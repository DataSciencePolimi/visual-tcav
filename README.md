from visual_tcav import LocalVisualTCAV, TorchModelWrapper
import torchvision.models as models

# Load ResNet50
resnet = models.resnet50(weights='DEFAULT')
wrapper = TorchModelWrapper(model_name="resnet50", model=resnet)

# See available layers
wrapper.info()

# Create explainer
tcav = LocalVisualTCAV(
    model_wrapper=wrapper,
    test_image_path="./zebra.jpg",
    concept_names=["striped", "dotted"],
    concept_base_dir="./concept_images",
    layer_names=["layer4"],
)

# Run
tcav.predict().info()
tcav.explain()
tcav.plot()