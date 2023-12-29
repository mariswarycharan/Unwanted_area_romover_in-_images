import os
import cv2


output_image_to_display = cv2.imread(r"C:\Users\iQube_VR\Documents\GPU setup python\pin_remover\output.png", cv2.IMREAD_UNCHANGED)

output_image_to_display = cv2.resize(output_image_to_display,(700,700))
trasn_mask = output_image_to_display[:,:,3 ]==0
output_image_to_display[trasn_mask]=[255, 255, 255, 255]
    

cv2.imshow("svuh",output_image_to_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

