from Architecture import MLP, CNN, CNN4, ResNet18, ResNet34, ResNet50, ResNet101,  VGG11, VGG16, AlexNet

def get_model(model_name, input_shape, num_classes):
    """
    Returns a model based on the specified model name.
    
    Args:
        model_name (str): Name of the model to retrieve.
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.
    
    Returns:
        torch.nn.Module: The requested model.
    """    
    if model_name == 'MLP':
        return MLP(input_shape, num_classes)
    elif model_name == 'CNN':
        return CNN(input_shape, num_classes)   
    elif model_name == 'CNN4':
        return CNN4(input_shape, num_classes)
    elif model_name == 'resnet18':
        return ResNet18(input_shape, num_classes)
    elif model_name == 'resnet34':
        return ResNet34(input_shape, num_classes)
    elif model_name == 'resnet50':
        return ResNet50(input_shape, num_classes)
    elif model_name == 'resnet101':
        return ResNet101(input_shape, num_classes)
    elif model_name == 'vgg11':
        return VGG11(input_shape, num_classes)
    elif model_name == 'vgg16':
        return VGG16(input_shape, num_classes)
    elif model_name == 'alexnet':
        return AlexNet(input_shape, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} is not supported.")