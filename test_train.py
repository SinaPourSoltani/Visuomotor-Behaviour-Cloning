import sys
import json
from datetime import datetime
from torchvision import transforms
sys.path.append('Visuomotor-Behaviour-Cloning')
from BehaviourCloningCNN import *

# With Shadows
data_link = "https://github.com/SinaPourSoltani/Visuomotor-Behaviour-Cloning/releases/download/v0.3/data.zip"
is_stereo = False

augmentations = None

load_data(data_link)
train_loader, valid_loader, test_loader = get_data_loaders(*get_episodes(),transforms=augmentations, is_stereo=is_stereo, std_noise_poke_vec=None)
model = get_model(is_stereo=is_stereo)

train_loss =  []
test_loss = []

model = freeze_backbone(model, is_stereo=is_stereo)
train_loss, tmp_train_acc, test_loss, tmp_test_acc = train(model,train_loader,valid_loader, lr=1e-4, max_epochs=60, patience=-1, is_stereo=is_stereo, model_tag="baseline")
plot_history(train_loss, tmp_train_acc, test_loss, tmp_test_acc)

time_stamp = datetime.now().strftime("%d-%H-%M")
filename = "ResNet18_" + time_stamp + ".pth"
torch.save(model.state_dict(), filename)

summary = {}
summary["Model name"] = filename;
summary["Epochs"] = len(train_loss)
summary["Train loss"] = train_loss
summary["Test loss"] = test_loss
summary["Is stereo"] = is_stereo

summary_file_name = "summary_" + time_stamp
with open(summary_file_name, 'w') as outfile:
            json.dump(summary, outfile)
