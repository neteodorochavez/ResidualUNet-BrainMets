def load_from_checkpoint(layer_dict):
    layer_dict_keys = list(layer_dict.keys())
    for key in layer_dict_keys:
        #print(f"Old Key: {key}")
        key_list = key.split('.')
        if key_list[0] == 'model':
            new_key = '.'.join(key_list[1:])
        else:
            new_key = '.'.join(key_list)
        layer_dict[new_key] = layer_dict.pop(key)
        #print(f"New Key: {new_key}")
    return layer_dict
