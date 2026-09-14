"""Verify TS_MASKS_PEDS alignment with THIN_LUNG_NII images.
Masks are 1.5x1.5x3.0mm with reset affine; images are native. Same FOV?
Test identity alignment vs axis flips using HU statistics inside organ labels.
"""
import sys, numpy as np, nibabel as nib
from scipy.ndimage import zoom

LUNG=[10,11,12,13,14]; HEART=[51,61]; AORTA=[52]; ESO=[15]

def resize_nn(a, out_shape):
    idx = [np.clip((np.arange(o)+0.5)*s/o-0.5, 0, s-1).round().astype(int)
           for s,o in zip(a.shape, out_shape)]
    return a[np.ix_(*idx)]

def main(accs):
    for acc in accs:
        im = nib.load(f'/temp_work/ch278233/BCH_DATASET/THIN_LUNG_NII/{acc}.nii')
        mk = nib.load(f'/temp_work/ch278233/TS_MASKS_PEDS/{acc}.nii.gz')
        izo = np.array(im.header.get_zooms()[:3], float)
        mzo = np.array(mk.header.get_zooms()[:3], float)
        ifov = np.array(im.shape[:3])*izo
        mfov = np.array(mk.shape[:3])*mzo
        img = np.asanyarray(im.dataobj).astype(np.float32)
        msk = np.asanyarray(mk.dataobj)
        # downsample image to mask grid (cheap: nearest index)
        imr = resize_nn(img, msk.shape)
        print(f'--- {acc} img{im.shape}{tuple(np.round(izo,3))} mask{msk.shape} '
              f'FOVimg{np.round(ifov,1)} FOVmask{np.round(mfov,1)} '
              f'FOVdiff{np.round(ifov-mfov,1)}')
        variants = {
            'identity': imr,
            'flipz': imr[:,:,::-1],
            'flipx': imr[::-1],
            'flipy': imr[:,::-1],
            'flipxy': imr[::-1,::-1],
        }
        for name,v in variants.items():
            out=[]
            for lab,ids in [('lung',LUNG),('heart',HEART),('aorta',AORTA)]:
                m = np.isin(msk, ids)
                out.append(f'{lab}={v[m].mean():7.1f}({m.sum()})' if m.sum() else f'{lab}=NA')
            print(f'   {name:9s} ' + '  '.join(out))

if __name__=='__main__':
    main(sys.argv[1:])
