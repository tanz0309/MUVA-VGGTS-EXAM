"""MUVA adapter using native O-MaMa matching, descriptors and adjacent negatives."""
import sys
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from source_audit import verify_source


def _image_tensor(image, size):
    import cv2
    from torchvision.transforms.functional import to_tensor, normalize
    image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return normalize(to_tensor(image.copy()), [.485, .456, .406], [.229, .224, .225])


def _resized_mask(mask, size):
    import cv2
    return torch.from_numpy(cv2.resize(np.asarray(mask).astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST))


class OmamaAdapter:
    def __init__(self, config):
        verify_source(config)
        self.config=config
        verify_source({'repo':config['fastsam_source'], 'repo_commit':config['fastsam_commit']})
        repo=Path(config['repo']); sys.path[:0]=[config['fastsam_source'],str(repo)]
        from fastsam import FastSAM
        from descriptors.get_descriptors import DescriptorExtractor
        from model.model import Attention_projector
        from unittest.mock import patch
        load = torch.hub.load
        def local_dino(repo, name, *args, **kwargs):
            if repo != 'facebookresearch/dinov2':
                raise ValueError('Unexpected Torch Hub repository')
            return load(config['dinov2_source'], name, source='local')
        with patch('torch.hub.load', side_effect=local_dino):
            self.extractor=DescriptorExtractor('dinov2_vitb14_reg',14,20,'cuda:0')
        self.extractor.model.requires_grad_(False).eval()
        self.fastsam=FastSAM(config['fastsam_checkpoint'])
        self.fastsam.model.cuda().eval().requires_grad_(False)
        self.predictor=Attention_projector(reverse=config['reverse']).cuda().eval()
        if config['checkpoint']:
            self.predictor.load_state_dict(torch.load(config['checkpoint'],map_location='cpu')['model'],strict=True)
        self.frozen_modules=[self.extractor.model,self.fastsam.model]

    @torch.no_grad()
    def prepare(self,reference,reference_mask,target):
        import cv2
        from dataset.dataset_utils import bbox_from_mask
        cfg=self.config; rs=tuple(cfg['reference_size']);ts=tuple(cfg['target_size'])
        # RGB input only; FastSAM's numpy interface expects BGR. No proposal disk cache in timed inference.
        results=self.fastsam(np.asarray(target)[:,:,::-1].copy(),device='cuda:0',
            retina_masks=True,imgsz=cfg['fastsam_image_size'],conf=cfg['fastsam_conf'],
            iou=cfg['fastsam_iou'],verbose=False,save=False)
        if results and results[0].masks is not None:
            masks=F.interpolate(results[0].masks.data[None].float(),size=ts,mode='nearest')[0].cpu().bool()
            masks=masks[masks.flatten(1).any(1)]
        else:
            masks=torch.empty((0,*ts),dtype=torch.bool)
        # Empty proposal set has a deterministic empty prediction, never a GT fallback.
        if len(masks)==0: masks=torch.zeros((1,*ts),dtype=torch.bool)
        boxes=torch.stack([bbox_from_mask(m)[0] for m in masks])
        sm=_resized_mask(np.asarray(reference_mask)>0,rs)
        source_box=bbox_from_mask(sm)[0]
        batch={'SOURCE_img':_image_tensor(np.asarray(reference),rs)[None],
            'SOURCE_mask':sm[None],'SOURCE_bbox':source_box[None],
            'SOURCE_img_size':torch.tensor([rs]),'GT_img':_image_tensor(np.asarray(target),ts)[None],
            'DEST_SAM_masks':masks[None],'DEST_SAM_bbox':boxes[None],
            'DEST_img_size':torch.tensor([ts])}
        return batch

    def descriptors(self,batch):
        dest,df=self.extractor.get_DEST_descriptors(batch)
        src,sf=self.extractor.get_SOURCE_descriptors(batch)
        return src,dest,sf,df

    @torch.no_grad()
    def predict(self,reference,reference_mask,target):
        self.predictor.eval(); batch=self.prepare(reference,reference_mask,target)
        scores,indices,pred,_,_=self.predictor(*self.descriptors(batch),None,None,batch['DEST_SAM_masks'],True)
        pred=F.interpolate(pred[None,None].float(),size=(target.height,target.width),mode='nearest')[0,0]
        return pred.cpu().numpy().astype(bool),{'proposal_count':int(batch['DEST_SAM_masks'].shape[1]),
            'selected_proposal':int(indices[0]),'matching_confidence':float(scores[0])}

    def _training_example(self,reference,reference_mask,target,load_target_gt):
        from scipy.spatial import QhullError
        from dataset.adj_descriptors import get_adj_matrix
        from dataset.dataset_masks import Masks_Dataset
        from dataset.dataset_utils import bbox_from_mask
        self.predictor.train(); batch=self.prepare(reference,reference_mask,target)
        gt=_resized_mask(load_target_gt(),tuple(self.config['target_size']))
        if not gt.any(): raise ValueError('Training GT vanished at O-MaMa target resolution')
        masks=batch['DEST_SAM_masks'][0]; boxes=batch['DEST_SAM_bbox'][0]
        fallback=False
        try: adj=get_adj_matrix(masks,order=2)
        except QhullError:
            # Collinear/duplicate proposal centers: use uniform remaining negatives.
            adj=torch.zeros(len(masks),len(masks)); fallback=True
        selector=object.__new__(Masks_Dataset)
        selector.N_masks_per_batch=self.config['training']['candidate_count']
        neg,neg_boxes=selector.select_adjacent_negatives(adj,boxes,masks,gt)
        pos=random.randrange(selector.N_masks_per_batch)
        batch['DEST_SAM_masks']=torch.cat([neg[:pos],gt[None],neg[pos:]])[None]
        gtbox=bbox_from_mask(gt)[0]
        batch['DEST_SAM_bbox']=torch.cat([neg_boxes[:pos],gtbox[None],neg_boxes[pos:]])[None]
        return batch,pos,{'training_candidates':selector.N_masks_per_batch,'positive_is_training_gt':True,
            'adjacency_fallback':fallback,'target_gt_pixels':int(gt.sum())}

    def training_loss(self,reference,reference_mask,target,load_target_gt):
        return self.training_loss_batch([(reference,reference_mask,target)],[load_target_gt])

    def training_loss_batch(self,inputs,load_target_gts):
        if not inputs or len(inputs)!=len(load_target_gts):
            raise ValueError('O-MaMa requires one training target loader per image pair')
        examples=[self._training_example(*sample,load_gt)
                  for sample,load_gt in zip(inputs,load_target_gts)]
        # Frozen image descriptors are independent per pair. Extract them serially
        # to bound backbone memory, then run the unchanged author's matching head
        # once on the complete minibatch and use its native mean-reduced loss.
        descriptors=[self.descriptors(batch) for batch,_,_ in examples]
        features=[torch.cat([item[index] for item in descriptors],dim=0)
                  for index in range(4)]
        masks=torch.cat([batch['DEST_SAM_masks'] for batch,_,_ in examples],dim=0)
        positions=torch.tensor([position for _,position,_ in examples])
        _,_,_,loss,_=self.predictor(*features,positions,torch.ones(len(examples)),masks,False)
        return loss,{'training_candidates':self.config['training']['candidate_count'],
            'positive_is_training_gt':True,'training_batch_size':len(examples),
            'adjacency_fallback':any(details['adjacency_fallback'] for _,_,details in examples),
            'target_gt_pixels':sum(details['target_gt_pixels'] for _,_,details in examples)}
