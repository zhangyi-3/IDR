import os, yaml
import copy
import operator

from utils import EasyDict

cfgs = {}


def dict2EdictRecursion(naive_dict):
    if type(naive_dict) != dict:
        return naive_dict

    easy_dict = EasyDict(naive_dict)
    for key in naive_dict.keys():  # transform all naive_dict to easy_dict
        if type(naive_dict[key]) == dict:
            easy_dict[key] = dict2EdictRecursion(easy_dict[key])
    return easy_dict


def yaml2edict(yaml_path):
    '''
    :input  yaml_path
    :return a complete easy_dict
    '''

    f = open(yaml_path)
    naive_dict = yaml.load(f, Loader=yaml.FullLoader)
    return dict2EdictRecursion(naive_dict)


def generateConfig(pre_name, cons, init_cfg):
    name = pre_name

    temp_cfg = copy.deepcopy(init_cfg)  #
    # temp_cfg = dict2EdictRecursion(temp_cfg)  # cannot copy the attribute properties. dict only

    temp_cfg.name = pre_name
    temp_cfg.model_dir = './results/%s' % name
    temp_cfg.result_dir = os.path.join(temp_cfg['model_dir'], 'res-%s' % name)

    for item in cons.keys():  # update dict
        if '.' in item:
            full_parts = item.split('.')
            part_name = '.'.join(full_parts[:-1])
            # print(temp_cfg.keys())
            # operator.attrgetter(part_name)(temp_cfg)[full_parts[-1]] = cons[item]
            # print('temp', operator.attrgetter('dataset.train')(temp_cfg)['batch_size_per_gpu'])
            try:
                operator.attrgetter(part_name)(temp_cfg)[full_parts[-1]] = cons[item]
            except:
                print('Cannot find key', item, part_name, full_parts[-1])
                exit()
        else:
            if item not in temp_cfg.keys() and item[:2] != 'tp':
                print('Cannot found the key', item)
                exit()
            temp_cfg[item] = cons[item]
    cfgs[name] = dict2EdictRecursion(temp_cfg)
    return


idr_config = yaml2edict('./configs/IDR.yaml')

name = 'idr-g'  # idr; Gaussian noise
generateConfig(name, {
    'noise_type': 'g',
    'noise_level': [0, 50],
}, idr_config)

name = 'idr-line'  # idr; line noise
generateConfig(name, {
    'noise_type': 'line',
    'noise_level': 5,
}, idr_config)

for item in ['binomial', 'impulse']:  # idr; binomial and impulse noise
    name = 'idr-' + item
    generateConfig(name, {
        'noise_type': item,
        'noise_level': 0.95,
    }, idr_config)

for item in range(1, 5):  # idr; spatially correlated noise 1~4
    name = 'idr-scn%d' % item
    generateConfig(name, {
        'noise_type': 'scn-%d' % item,
        'noise_level': 5,
    }, idr_config)

name = 'n2n-g'  # n2n; Gaussian noise
generateConfig(name, {
    'noise_type': 'g',
    'trainer.name': 'DenoiseBase_n2n',
}, idr_config)

name = 'n2c-g'  # n2c; Gaussian noise
generateConfig(name, {
    'noise_type': 'g',
    'trainer.name': 'DenoiseBase_n2c',
}, idr_config)
