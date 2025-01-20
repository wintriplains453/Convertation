from runners.simple_runner import SimpleRunner

runner = SimpleRunner(editor_ckpt_pth="pretrained_models/sfe_editor_light.pt")

# Выполнение редактирования
edited_image_path = f"notebook/images/gosling.jpg"
neutral_prompt: str = "face"
target_prompt: str = "hair_face"
disentanglement: float = 0.18

runner.edit(
    orig_img_pth="notebook/images/smith.jpg",
    editing_name="age",
    edited_power=5,
    save_pth="editing_res/smith/smith.jpg",
    align=True,
    save_inversion=True
)