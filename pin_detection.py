from ultralytics import YOLO
import torch,shutil
import cv2

input = cv2.imread(r"C:\Users\iQube_VR\Documents\GPU setup python\pin_remover\28335-CLOWN FISH RUBBER-9\28335-CLOWN FISH RUBBER-9-49.jpg")
input = cv2.resize(input,(700,700))



model = YOLO(r'C:\Users\iQube_VR\Documents\GPU setup python\pin_remover\models\best.pt')
results = model.predict(source=input,device=0,save=True)

for result in results:
    # get array results
    masks = result.masks.masks
    boxes = result.boxes.boxes
    # extract classes
    clss = boxes[:, 5]
    # get indices of results where class is 0 (people in COCO)
    people_indices = torch.where(clss == 0)
    # use these indices to extract the relevant masks
    people_masks = masks[people_indices]
    # scale for visualizing results
    people_mask = torch.any(people_masks, dim=0).int() * 255
   
    # save to file
    cv2.imwrite('merged_images.png', people_mask.cpu().numpy())
    
    
cv2.imshow("predict",cv2.imread(r"C:\Users\iQube_VR\Documents\GPU setup python\pin_remover\runs\segment\predict\image0.jpg"))

mask = cv2.imread(r"C:\Users\iQube_VR\Documents\GPU setup python\pin_remover\merged_images.png",0)
mask = cv2.resize(mask,(700,700))

mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))

cv2.imshow("mask",mask)
cv2.imwrite("merged_images.png",mask)


shutil.rmtree(r"C:\Users\iQube_VR\Documents\GPU setup python\pin_remover\runs")

cv2.waitKey(0)
cv2.destroyAllWindows()