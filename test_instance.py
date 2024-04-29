image = "/path/to/an/imgage.jpg"
im = cv2.imread(image)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
               metadata=leaf_metadata, 
               scale=0.8, 
               instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

plt.figure(figsize = (14, 10))
plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
plt.show()