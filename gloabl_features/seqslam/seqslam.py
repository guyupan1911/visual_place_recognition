import cv2
import numpy as np

class SeqSLAM():
  
    def run(self, image_path: str):
        # load image and resize
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))

        # to numpy
        image = np.copy(np.asarray(image))

        # normalization
        feature = self.patchNormalize(image)

        return feature

    def patchNormalize(self, image):
        n = range(0, image.shape[0]+2, 8)
        m = range(0, image.shape[1]+2, 8)
            
        for i in range(len(n)-1):
            for j in range(len(m)-1):
                p = image[n[i]:n[i+1], m[j]:m[j+1]]
                pp=np.copy(p.flatten('c'))
                pp=pp.astype(float)
                image[n[i]:n[i+1], m[j]:m[j+1]] = 127+np.reshape(np.round((pp-np.mean(pp))/np.std(pp, ddof=1)), (8, 8))
                    
        return image.flatten('c')
