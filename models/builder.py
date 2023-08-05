def process_model(model_name, input_shape, num_classes, seed):
  if model_name == 'mobilenetv2_siamese':
    from models.mobilenetv2_siamese import get_model
  elif model_name == 'mobilenetv2':
    from models.mobilenetv2 import get_model

  if model_name == 'efficientnetb0_siamese':
    from models.efficientnetb0_siamese import get_model
  elif model_name == 'efficientnetb0':
    from models.efficientnetb0 import get_model

  if model_name == 'inceptionv3_siamese':
    from models.inceptionv3_siamese import get_model
  elif model_name == 'inceptionv3':
    from models.inceptionv3 import get_model
    
  if model_name == 'resnet50_siamese':
    from models.resnet50_siamese import get_model
  elif model_name == 'resnet50':
    from models.resnet50 import get_model

  return get_model(input_shape, num_classes, seed)
