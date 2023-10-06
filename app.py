import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from fastsam import FastSAM, FastSAMPrompt

project_path = "/home/ravi.naik/learning/era/s19"
sam_model = FastSAM(f"{project_path}/FastSAM-x.pt")

DEVICE = "cpu"
sample_images = [f"{project_path}/sample_images/{i}.jpg" for i in range(5)]
prediction_image = None
sam_prediction_image = None

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


def read_image(path):
    img = Image.open(path)
    return img


def set_prediction_image(evt: gr.SelectData, gallery):
    global prediction_image
    if isinstance(gallery[evt.index], dict):
        prediction_image = gallery[evt.index]["name"]
    else:
        prediction_image = gallery[evt.index][0]["name"]


def predict(text):
    text_classes = text.split(",")
    text_classes = [sentence.strip() for sentence in text_classes]

    image = read_image(prediction_image)

    inputs = clip_processor(
        text=text_classes,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    outputs = clip_model(**inputs)
    logits_per_image = (
        outputs.logits_per_image
    )  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)[0]
    results = {text_class: prob.item() for text_class, prob in zip(text_classes, probs)}
    return {output: gr.update(value=results)}


def show_hide_sam_text(status):
    if status == "Text Based":
        return {sam_input_text: gr.update(visible=True)}
    return {sam_input_text: gr.update(visible=False)}


def set_prediction_image_sam(evt: gr.SelectData, gallery):
    global sam_prediction_image
    if isinstance(gallery[evt.index], dict):
        sam_prediction_image = gallery[evt.index]["name"]
    else:
        sam_prediction_image = gallery[evt.index][0]["name"]


def sam_predict(radio, text):
    output_path = f"{project_path}/output/sam_results.jpg"
    everything_results = sam_model(
        sam_prediction_image,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    prompt_process = FastSAMPrompt(
        sam_prediction_image, everything_results, device=DEVICE
    )
    ann = prompt_process.everything_prompt()
    if radio == "Text Based":
        ann = prompt_process.text_prompt(text=text)

    prompt_process.plot(
        annotations=ann,
        output_path=output_path,
    )

    return {sam_output: gr.update(value=output_path)}


with gr.Blocks() as app:
    gr.Markdown("## ERA Session19 - FastSAM & CLIP Inference with Gradio")
    with gr.Tab("FastSAM"):
        gr.Markdown("### ERA Session19 - Image Segmentation with FastSAM")
        gr.Markdown(
            """Please an image or select one of the sample images.
            Select either segment everything or text based segmentation.
            Enter the text if you opt for segment based on text and hit Submit.
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        upload_gallery = gr.Gallery(
                            value=None,
                            label="Uploaded images",
                            show_label=False,
                            elem_id="gallery_upload",
                            columns=5,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                        )
                        upload_button = gr.UploadButton(
                            "Click to Upload images",
                            file_types=["image"],
                            file_count="multiple",
                        )
                        upload_button.upload(upload_file, upload_button, upload_gallery)

                    with gr.Group():
                        sample_gallery = gr.Gallery(
                            value=sample_images,
                            label="Sample images",
                            show_label=False,
                            elem_id="gallery_sample",
                            columns=3,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                        )

                    upload_gallery.select(
                        set_prediction_image_sam, inputs=[upload_gallery]
                    )
                    sample_gallery.select(
                        set_prediction_image_sam, inputs=[sample_gallery]
                    )
                with gr.Box():
                    radio = gr.Radio(
                        choices=["Segment Everything", "Text Based"],
                        value="Segment Everything",
                        type="value",
                        label="Select a Segmentation approach",
                        interactive=True,
                    )
                    sam_input_text = gr.TextArea(
                        label="Segementation Input",
                        placeholder="Please enter some text",
                        interactive=True,
                        visible=False,
                    )
                radio.change(
                    show_hide_sam_text, inputs=[radio], outputs=[sam_input_text]
                )

                sam_submit_btn = gr.Button(value="Submit")
            with gr.Column():
                with gr.Box():
                    sam_output = gr.Image(value=None, label="Segmentation Results")

                sam_submit_btn.click(
                    sam_predict, inputs=[radio, sam_input_text], outputs=[sam_output]
                )
    with gr.Tab("CLIP"):
        gr.Markdown("### ERA Session19 - Zero Shot Classification with CLIP")
        gr.Markdown(
            "Please an image or select one of the sample images. Type some classification labels separated by comma. For ex: dog, cat"
        )
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    with gr.Group():
                        upload_gallery = gr.Gallery(
                            value=None,
                            label="Uploaded images",
                            show_label=False,
                            elem_id="gallery_upload",
                            columns=5,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                        )
                        upload_button = gr.UploadButton(
                            "Click to Upload images",
                            file_types=["image"],
                            file_count="multiple",
                        )
                        upload_button.upload(upload_file, upload_button, upload_gallery)

                    with gr.Group():
                        sample_gallery = gr.Gallery(
                            value=sample_images,
                            label="Sample images",
                            show_label=False,
                            elem_id="gallery_sample",
                            columns=3,
                            rows=2,
                            height="auto",
                            object_fit="contain",
                        )

                    upload_gallery.select(set_prediction_image, inputs=[upload_gallery])
                    sample_gallery.select(set_prediction_image, inputs=[sample_gallery])
                with gr.Box():
                    input_text = gr.TextArea(
                        label="Classification Text",
                        placeholder="Please enter comma separated text",
                        interactive=True,
                    )

                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                with gr.Box():
                    output = gr.Label(value=None, label="Classification Results")

                submit_btn.click(predict, inputs=[input_text], outputs=[output])


app.launch(debug=True, show_error=True)
