import TMAx

path = r"D:\imvia\unet_tma_segmenter\npX2EednKVIOEsp1fhzIQLEYcCF.jpg"
out = r"D:\imvia\unet_tma_segmenter\npX2EednKVIOEsp1fhzIQLEYcCF_out.jpg"
i = TMAx.predict_mask(path, out)
