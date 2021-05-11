from BehaviourCloningCNN import *

#data_link = "https://github.com/SinaPourSoltani/Visuomotor-Behaviour-Cloning/releases/download/v0.3/data.zip"
#load_data(data_link)
train_loader, valid_loader, test_loader = get_data_loaders(*get_episodes(), is_stereo=True)
model = get_model()

plot_history(*train(model,train_loader,valid_loader, lr=1e-4, max_epochs=3, patience=30, is_stereo=True))

test_acc = one_epoch(model, test_loader)[1]
print(f'{test_acc * 100:.1f} % test accuracy')

filename = "ResNet18_Neps" + str(len(get_episodes()[0])) + ".pth"
torch.save(model.state_dict(), filename)
