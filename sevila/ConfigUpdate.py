import re
from lavis.common.config import Config
from omegaconf import OmegaConf

# 用于递归地替换字典、列表或字符串中的值
def replace_in_config(config, old_pattern, new_value):
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = replace_in_config(value, old_pattern, new_value)
    elif isinstance(config, list):
        for index, item in enumerate(config):
            config[index] = replace_in_config(item, old_pattern, new_value)
    elif isinstance(config, str):
        return re.sub(old_pattern, new_value, config)
    return config



def update_config(cfg, id_value):
    import re
    from omegaconf import OmegaConf

    # 将 Config 对象的内部配置转换为 Python 字典
    config_dict = OmegaConf.to_container(cfg.config, resolve=False)

    # 替换所有匹配的 "${id}" 字符串
    updated_dict = replace_in_config(config_dict, r'^\d+$', str(id_value))

    # 将更新后的字典转换回 OmegaConf 配置对象
    updated_omegaconf = OmegaConf.create(updated_dict)
    print("updated_omegaconf:::")
    print(updated_omegaconf)

    # 确保 `args.options` 已被正确转换为字典格式
    if isinstance(cfg.args.options, list):
        options_dict = {}
        for opt in cfg.args.options:
            key, value = opt.split('=', 1)  # 将选项分割为键值对
            options_dict[key] = value
        cfg.args.options = options_dict  # 更新为字典

    # 确保 `options` 键在配置中存在
    if 'options' not in cfg.args:
        cfg.args.options = {}

    # 构造新的 Config 对象
    updated_cfg = Config(updated_omegaconf)

    return updated_cfg

