import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from test_models import testing
from Adversarial_attack import pgd

def truncated_normal(size, epsilon):
    values = torch.fmod(torch.randn(size), 2) * epsilon
    return values

def label_smooth(y, num_classes, weight=0.9):
    # requires y to be one_hot!
    return F.one_hot(y, num_classes=num_classes).type(torch.float).clamp(min=(1. - weight) / (num_classes - 1.), max=weight)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
    
def random_flip_left_right(images):
    images_flipped = torch.flip(images, dims=[3])
    flip = torch.bernoulli(torch.ones(images.shape[0],) * 0.5).type(torch.bool)
    images[flip] = images_flipped[flip]
    return images.detach()

# Learning rate schedule for training
def lr_schedule(total_epoches, epoch, lr_max):
    if total_epoches >= 110:
        if epoch / total_epoches < 0.5:
            return lr_max
        elif epoch / total_epoches < 0.75:
            return lr_max / 10.
        elif epoch / total_epoches < 0.9:
            return lr_max / 100.
        else:
            return lr_max / 200.
        
def training(train_loader, test_loader, model, args, device=None):
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MOMENTUM = args.momentum
    WEIGHT_DECAY = args.weight_decay
    START_EPOCH = args.start_epoch
    TRAIN_METHOD = args.training_method
    TEST_FREQUENCY = args.test_frequency
    
    best_accuracy = 0.
    best_prec_history = 0.
    best_accu_robust = 0.
    best_prec_robust_history = 0.
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    for epoch in range(START_EPOCH, EPOCHS):
        if TRAIN_METHOD == 'standard':
            train_standard(train_loader, model, optimizer, LEARNING_RATE, epoch, EPOCHS, device)
        elif TRAIN_METHOD == 'pgd':
            train_pgd(train_loader, model, optimizer, LEARNING_RATE, epoch, EPOCHS, args, device)
        elif TRAIN_METHOD == 'GAIRAT':
            train_GAIRAT(train_loader, model, optimizer, max_lr, epoch, num_epochs, begin_epoch, device)
        
        accuracy = testing(test_loader, model, step_size=0., eps=0., device=device)
        print ('Acc: {:.3f}'.format(accuracy))
        best_accuracy = max(accuracy, best_accuracy)
        if best_accuracy > best_prec_history:
            best_prec_history = best_accuracy
            save_checkpoint(model.state_dict(), filename='Model_{}.ckpt'.format(TRAIN_METHOD))
        
        if TRAIN_METHOD == 'pgd' or TRAIN_METHOD == 'GAIRAT':
            # Set adversarial training parameters
            EPSILON = args.epsilon
            STEP_SIZE = args.step_size         
            accu_adversarial = testing(test_loader, model, STEP_SIZE, eps=EPSILON, attack='pgd', device=device)
            print ('Acc adversarial: {:.3f}'.format(accu_adversarial))
            best_accu_robust = max(accu_adversarial, best_accu_robust)
            if best_accu_robust > best_prec_robust_history:
                best_prec_robust_history = best_accu_robust
                save_checkpoint(model.state_dict(), filename='Model_robust_{}.ckpt'.format(TRAIN_METHOD))

def train_standard(train_loader, model, optimizer, max_lr, epoch, num_epochs, device):
    train_loss = 0
    train_correct = 0
    train_total = 0
    total_step = len(train_loader)
    model.train()
    
    # Set learning rate
    lr = lr_schedule(num_epochs, epoch + 1, max_lr)
    optimizer.param_groups[0].update(lr=lr)
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        images = random_flip_left_right(images)
        outputs = model(images)
        loss = nn.CrossEntropyLoss(reduction="mean")(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], LR: {:.4f}, Loss: {:.4f}, Acc: {:.3f}' 
                               .format(epoch+1, num_epochs, i+1, total_step,
                                       optimizer.param_groups[0]['lr'], train_loss/total_step, 100.*train_correct/train_total))        

# Adversarial training with PGD
# 'madry2017towards'
# 'Towards deep learning models resistant to adversarial attacks'
def train_pgd(train_loader, model, optimizer, max_lr, epoch, num_epochs, args, device):
    train_loss = 0
    train_correct = 0
    train_total = 0
    total_step = len(train_loader)
    model.train()
    
    # Set learning rate
    lr = lr_schedule(num_epochs, epoch + 1, max_lr)
    optimizer.param_groups[0].update(lr=lr)
    # Set adversarial training parameters
    EPSILON = args.epsilon
    STEP_SIZE = args.step_size
    # NUM_STEPS = args.num_steps
    
    for i, (images, labels) in enumerate(train_loader):
        eps_defense = truncated_normal(images.shape[0], EPSILON).abs()
        eps_defense = eps_defense[:, None, None, None]
        eps_defense = eps_defense.to(device)
        num_half_images = int(images.shape[0] / 2)
        images, labels = images.to(device), labels.to(device)
        images_flipped = random_flip_left_right(images)
        outputs = model(images_flipped)
        _, predicted_label = outputs.max(1)
        random_number = truncated_normal(1, threshold=8.).abs()
        NUM_STEPS = int(min(random_number + 4, 1.25 * random_number))
        
        images_adv, _ = pgd(model, images_flipped, predicted_label, eps_defense, NUM_STEPS,
                           STEP_SIZE, rand_init=False, device=device)
        images_combined = torch.cat((images_flipped[0:num_half_images], images_adv[num_half_images:]), dim=0)
        
        outputs_worse = model(images_combined)
        loss_1 = nn.CrossEntropyLoss(reduction="mean")(outputs_worse[0:num_half_images], labels[0:num_half_images]) * 2.0 / 1.3
        loss_2 = nn.CrossEntropyLoss(reduction="mean")(outputs_worse[num_half_images:], labels[num_half_images:]) * 0.6 / 1.3
        loss = (loss_1 + loss_2) / 2.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs_worse.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], LR: {:.4f}, Loss: {:.4f}, Acc: {:.3f}' 
                               .format(epoch+1, num_epochs, i+1, total_step,
                                       optimizer.param_groups[0]['lr'], train_loss/total_step, 100.*train_correct/train_total))

# Adversarial training with GAIRAT
# 'zhang2020geometry'
# 'geometry aware instance reweighted adversarial training'

# Compute weighting on each sample based on Kappa
def GAIR(num_steps, Kappa, Lambda, func):
    # Weight assign
    if func == "Tanh":
        reweight = ((Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).tanh()+1)/2
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Sigmoid":
        reweight = (Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).sigmoid()
        normalized_reweight = reweight * len(reweight) / reweight.sum()
    elif func == "Discrete":
        reweight = ((num_steps+1)-Kappa)/(num_steps+1)
        normalized_reweight = reweight * len(reweight) / reweight.sum()
            
    return normalized_reweight

# Adjust lambda for weight assignment using epoch
def adjust_Lambda(epoch, epoches, Lambda, Lambda_max, Lambda_schedule):
    Lam = float(Lambda)
    if epoches >= 110:
        # Train Wide-ResNet
        Lambda = Lambda_max
        if Lambda_schedule == 'linear':
            if epoch >= 60:
                Lambda = Lambda_max - (epoch/args.epochs) * (Lambda_max - Lam)
        elif Lambda_schedule == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam-1.0
            elif epoch >= 110:
                Lambda = Lam-1.5
        elif Lambda_schedule == 'fixed':
            if epoch >= 60:
                Lambda = Lam
    else:
        # Train ResNet
        Lambda = Lambda_max
        if Lambda_schedule == 'linear':
            if epoch >= 30:
                Lambda = Lambda_max - (epoch/args.epochs) * (Lambda_max - Lam)
        elif Lambda_schedule == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam-2.0
        elif Lambda_schedule == 'fixed':
            if epoch >= 30:
                Lambda = Lam
    return Lambda

def train_GAIRAT(train_loader, model, optimizer, max_lr, epoch, num_epochs, args, device):
    train_loss = 0
    train_correct = 0
    train_total = 0
    total_step = len(train_loader)
    model.train()
    
    # Set learning rate
    lr = lr_schedule(num_epochs, epoch + 1, max_lr)
    optimizer.param_groups[0].update(lr=lr)
    # Set adversarial training parameters
    EPSILON = args.epsilon
    STEP_SIZE = args.step_size
    NUM_STEPS = args.num_steps    
    WEIGHT_ASSIGNMENT_FUNCTION = args.weight_assignment_function
    
    # Get lambda
    LAMBDA = args.Lambda
    LAMBDA_MAX = args.Lambda_max
    LAMBDA_SCHEDULE = args.Lambda_schedule
    Lambda = adjust_Lambda(num_epochs, epoch + 1, LAMBDA, LAMBDA_MAX, LAMBDA_SCHEDULE)
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        lr = lr_schedule(num_epochs, epoch + 1, max_lr)
        optimizer.param_groups[0].update(lr=lr)
        
        x_adv, Kappa = pgd(model, data, target, EPSILON, NUM_STEPS,
                           STEP_SIZE, rand_init=True, device=None)
        logit = model(x_adv)
        _, predicted_label = outputs.max(1)
        if (epoch + 1) >= BEGIN_EPOCH:
            Kappa = Kappa.to(device)
            loss = nn.CrossEntropyLoss(reduction=False)(logit, target)
            # Calculate weight assignment according to geometry value
            normalized_reweight = GAIR(NUM_STEPS, Kappa, Lambda, WEIGHT_ASSIGNMENT_FUNCTION)
            loss = loss.mul(normalized_reweight).mean()            
        else:
            loss = nn.CrossEntropyLoss(reduction="mean")(logit, target)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
    
        train_loss += loss.item()
        _, predicted = outputs_worse.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], LR: {:.4f}, Loss: {:.4f}, Acc: {:.3f}' 
                               .format(epoch+1, num_epochs, i+1, total_step,
                                       optimizer.param_groups[0]['lr'], train_loss/total_step, 100.*train_correct/train_total))    
