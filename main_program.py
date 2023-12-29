

import rembg
import shutil
import cv2
import numpy as np
import torch
import PIL
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline

def load_image(path, size=(512, 512)):
    with PIL.Image.open(path) as image:
        return image.resize(size)

def detect_objects(model, image):
    results = model.predict(source=image, device=0, save=True)
    return results

def process_masked_area(results):
    processed_mask = None
    if str(results[0].masks) != "None":
        for result in results:
            masks = result.masks.data
            boxes = result.boxes.data
            clss = boxes[:, 5]
            people_indices = torch.where(clss == 0)
            people_masks = masks[people_indices]
            people_mask = torch.any(people_masks, dim=0).int() * 255
            cv2.imwrite('merged_images.png', people_mask.cpu().numpy())
        
        mask = cv2.imread('merged_images.png', 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))
        cv2.imwrite("merged_images.png", mask)

        mask_image = PIL.Image.open('merged_images.png').resize((512, 512))
        processed_mask = np.array(mask_image)

    return processed_mask

def inpaint_image(pipe, prompt, image, mask, guidance_scale, generator, num_samples):
    images = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    images[0].save("out_inpaint.png")

    inpainted_image = rembg.remove(cv2.imread('out_inpaint.png'), bgcolor=(0, 0, 0, 0))
    cv2.imwrite("final_output_last_fill_white_bg.png", inpainted_image)

def display_images(input_image, final_output):
    cv2.imshow("input_image", input_image)
    cv2.imshow("final_output_without_bg", cv2.imread("final_output_last_fill_white_bg.png"))
    cv2.imshow("final_output_filled_with_white_bg", final_output)

def main():
    device = "cuda"
    model_path = "stabilityai/stable-diffusion-2-inpainting"
    model = YOLO(r'models\best.pt')

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    prompt = "Restructure the masked area to correspond with the colors of neighboring pixels."
    guidance_scale = 7.5
    num_samples = 1
    generator = torch.Generator(device="cuda").manual_seed(0)

    input_path = r"28335-CLOWN FISH RUBBER-9\28335-CLOWN FISH RUBBER-9-12.jpg"
    input_image = cv2.resize(cv2.imread(input_path), (700, 700))

    results = detect_objects(model, input_image)
    processed_mask = process_masked_area(results)

    if processed_mask is not None:
        inpaint_image(pipe, prompt, load_image(input_path), processed_mask, guidance_scale, generator, num_samples)
        final_output = cv2.imread('final_output_last_fill_white_bg.png', cv2.IMREAD_UNCHANGED)
        final_output = cv2.resize(final_output, (700, 700))
        transparent_mask = final_output[:, :, 3] == 0
        final_output[transparent_mask] = [255, 255, 255, 255]
        display_images(input_image, final_output)
    else:
        cv2.imshow("input", input_image)
        cv2.imshow("yolo_not_detected_final_output", rembg.remove(input_image, bgcolor=(0, 0, 0, 0)))

    shutil.rmtree(r"runs")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
