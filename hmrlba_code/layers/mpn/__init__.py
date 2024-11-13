from hmrlba_code.layers.mpn.wln import WLNConv, WLNResConv, WLNConvAtt

WLNs = {'wln': WLNConv, 'wlnres': WLNResConv, 'wlnatt': WLNConvAtt}


def mpn_layer_from_config(config, encoder):
    if encoder in WLNs:
        layer_class = WLNs.get(encoder)
        mpn_layer = layer_class(node_fdim=config['node_fdim'],
                                edge_fdim=config['edge_fdim'],
                                hsize=config['hsize'],
                                depth=config['depth'],
                                dropout=config['dropout_p'],
                                activation=config['activation'],
                                jk_pool=config.get("jk_pool", None))

    elif encoder == 'gtrans':
        raise NotImplementedError()

    else:
        raise ValueError(f"Encoder {encoder} is not supported yet.")
    return mpn_layer
