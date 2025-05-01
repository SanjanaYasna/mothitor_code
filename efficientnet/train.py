from torchvision import datasets, transforms
import torch 
import torchvision.models as models
import torch.nn as nn
#Import confusion amtrix
from  torchmetrics import ConfusionMatrix 
import csv 

#training args TODO
batch_size = 500  
log_interval = 33  #for init every 20 train batches, but for all data, every 33 (so both 3x per epoch)
num_epochs = 50 
model_save = "/work/pi_mrobson_smith_edu/mothitor/binary_dataset/efficientnet/total/model_tensors/"
train_csv = "/work/pi_mrobson_smith_edu/mothitor/binary_dataset/efficientnet/total/log/train.csv"
test_csv = "/work/pi_mrobson_smith_edu/mothitor/binary_dataset/efficientnet/total/log/test.csv" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#write header as log_count, epoch, loss
with open(train_csv, 'a') as train_log:
    #write headers
    train_log.write("log_count, epoch, loss\n")
    train_log.flush()
train_log.close()
with open(test_csv, 'a') as test_log:
    test_log.write("log_count, epoch, test_loss, tn, fp, fn, tp\n")
    test_log.flush()
test_log.close()
    

#create transform function such that it resizes images to mean width and height of batch
transform = transforms.Compose([
    transforms.Resize((90, 110)),
    transforms.ToTensor(),
])


#train loader
dataset = datasets.ImageFolder('/work/pi_mrobson_smith_edu/mothitor/binary_dataset/train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

#test loader
test_dataset = datasets.ImageFolder('/work/pi_mrobson_smith_edu/mothitor/binary_dataset/test', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

assert test_dataset.class_to_idx == dataset.class_to_idx, "Train and test datasets must have the same class indices"
print("Class mapping: ", dataset.class_to_idx)

#onlly 2 classes, so change final classification head out dimension
model = models.efficientnet_b0(weights='DEFAULT')
model.classifier[1] = nn.Linear(in_features=1280, out_features=2).to(device)
model = model.to(device)
#assert model is on cuda
assert next(model.parameters()).is_cuda, "Model is not on GPU"

#cosine annealing schedule
#adam optimizer
total_steps = len(dataloader) * num_epochs
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0001)

#for fine-tuning, unfreeze gradients 
for params in model.parameters():
    params.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")

def log_model_tensors(model, log_num, epoch, optimizer, scheduler):
    #save model weights with log_num
    torch.save(model.state_dict(), model_save + f"epoch_{epoch}_{log_num}_tensors")
    #save optimizer and scheduler
    # torch.save(optimizer.state_dict(), model_save + f"_{log_num}_optimizer")
    # torch.save(scheduler.state_dict(), model_save + f"_{log_num}_scheduler")

#get test loss, and also confusion matrix of predictions 
def calculate_test_metrics(model, log_count, epoch):
    model.eval() 
    test_loss_acc = 0
    #confusion matrix
    confusion_matrix = ConfusionMatrix(task="binary", num_classes=2).cuda()
    confusion_matrix_total = torch.zeros(2, 2).cuda()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        # Forward pass
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        
        #get confusion matrix between output and target
        pred = output.argmax(dim=1)
        confusion = confusion_matrix(pred, target) 
        #add to confusion_matrix_total 
        confusion_matrix_total += confusion
        #add loss to test loss
        test_loss_acc += loss.item()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #at end, get test loss
    test_loss = test_loss_acc / len(test_loader)
    tn = confusion_matrix_total[0, 0]
    fp = confusion_matrix_total[0, 1]
    fn = confusion_matrix_total[1, 0]
    tp = confusion_matrix_total[1, 1]
    #log
    with open(test_csv, 'a') as test_log:
        test_log.write(f"{log_count}, {epoch}, {test_loss}, {tn}, {fp}, {fn}, {tp}\n") 
        test_log.flush()
    test_log.close()
    
#train function , CALLS TEST at right log intervals 
def train(model, train_loader, optimizer, schduler, epoch):
    train_loss_acc =0
    log_count = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        # Forward pass
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        #add loss to train loss
        train_loss_acc += loss.item()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() 

        if batch_idx % log_interval == 0:
            #get train and test loss
            
            #train loss is just train_loss_acc / log_inteval
            train_loss = train_loss_acc / log_interval
            #log train loss in csv and flush out output
            with open(train_csv, 'a') as train_log:
                train_log.write(f"{log_count}, {epoch}, {train_loss}\n")
                train_log.flush()
            train_log.close()
            #set train loss back to 0
            train_loss_acc = 0
             
            #now get test loss and metrics 
            calculate_test_metrics(model, log_count, epoch)
            log_count += 1

#call train
for epoch in range(num_epochs):
    train(model, dataloader, optimizer, scheduler, epoch)
    #save model weights with epoch number
    # log_model_tensors(model, log_count, epoch, optimizer, scheduler)
    torch.save(model.state_dict(), model_save + f"epoch_{epoch}_tensors")