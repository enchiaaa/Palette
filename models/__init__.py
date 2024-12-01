from core.praser import init_obj

def create_model(**cfg_model):
    """ create_model """
    opt = cfg_model['opt']
    logger = cfg_model['logger']

    model_opt = opt['model']['which_model']
    model_opt['args'].update(cfg_model)
    model = init_obj(model_opt, logger, default_file_name='models.model', init_type='Model')

    return model

def define_network(logger, opt, network_opt):
    """ define network with weights initialization """
    net = init_obj(network_opt, logger, default_file_name='models.network', init_type='Network')

    if opt['phase'] == 'train':
        logger.info('Network [{}] weights initialize using [{:s}] method.'.format(net.__class__.__name__, network_opt['args'].get('init_type', 'default')))
        net.init_weights()
    return net


# 根據配置 loss_opt 創建損失函數，logger 用於紀錄
def define_loss(logger, loss_opt):
    return init_obj(loss_opt, logger, default_file_name='models.loss', init_type='Loss')

# 根據配置 metric_opt 創建指標，logger 用於紀錄
def define_metric(logger, metric_opt):
    return init_obj(metric_opt, logger, default_file_name='models.metric', init_type='Metric')

