import torch
import utils
from imagenet import get_x_y_from_data_dict
import torchmetrics 


def validate(val_loader, model, criterion, args):
    """
    Run evaluation
    """
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    device = (torch.device("cuda:%s"%args.gpu) if torch.cuda.is_available() else torch.device("cpu"))
    
    # accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes+(1 if args.unlearn=='boundary_expanding' else 0), average="none").to(device)
    
    # commenting the above line and adding the below line for chestmnist
    if args.dataset == "chestmnist":
        accuracy_metric = None
    else:
        accuracy_metric = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=args.num_classes + (
                1 if args.unlearn=='boundary_expanding' else 0
            ),
            average="none"
        ).to(device)
    
    
    
    # switch to evaluate mode
    model.eval()
    if args.imagenet_arch:
        for i, data in enumerate(val_loader):
            image, target = get_x_y_from_data_dict(data, device)
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
           
            # accuracy_metric.update(output,target)

            # commenting above line and added the below line for the chesrtmnist
            if args.dataset != "chestmnist":
                accuracy_metric.update(output,target)



            # prec1 = utils.accuracy(output.data, target)[0]

            # commenting above line and adding the below line of code for chestmnist
            if args.dataset == "chestmnist":
                prec1 = utils.multilabel_accuracy(output.data, target)
            else:
                prec1 = utils.accuracy(output.data, target)[0]


            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        # print("args.num_classes:",args.num_classes)
        for i, (image, target) in enumerate(val_loader):
            image = image.to(device)
            target = target.to(device)

            # compute output
            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss

            # accuracy_metric.update(output,target)

            # commenting the above line and adding the below line for chestmnist
            if args.dataset != "chestmnist":
                accuracy_metric.update(output,target)


            # prec1 = utils.accuracy(output.data, target)[0]

            # commenting above line and adding the below code 
            if args.dataset == "chestmnist":
                prec1 = utils.multilabel_accuracy(output.data, target)
            else:
                prec1 = utils.accuracy(output.data, target)[0]


            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if i % args.print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})".format(
                        i, len(val_loader), loss=losses, top1=top1
                    )
                )

        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))
    
    # per_class_accuracies = accuracy_metric.compute().cpu().numpy()
    
    # commenting the above line and adding the below line for chestmnist
    if args.dataset != "chestmnist":
        per_class_accuracies = accuracy_metric.compute().cpu().numpy()
    else:
        per_class_accuracies = None
        
    
    
    # print("per_class_accuracies:",per_class_accuracies)
    return top1.avg,per_class_accuracies
