def get_ident_str(type, config, **kwargs):
    if type == 'l96': 
        f_ident = (
            f"c{config['c']}_"
            f"dt{config['dt']}_"
            f"si{config['si']}_"
            f"time{config['t_total']}_"
            f"rs{config['seed']}"
        )
        
    elif type == 'gcm':
        f_ident = (
            f"c{config['c']}_"
            f"dt{config['dt']}_"
            f"si{config['si']}_"
            f"time{config['t_total']}_"
            f"rs{config['seed']}"
        )
        
    elif type == 'ensemble': 
        seeds = kwargs['seeds']
        seed_str = f"{seeds[0]}-{seeds[-1]}"
        
        f_ident = (
            f"c{config['c']}_"
            f"dt{config['dt']}_"
            f"si{config['si']}_"
            f"time{config['t_total']}_"
            f"init{config['n_init_states']}_"
            f"ens{config['n_ens']}_"
            f"rs{seed_str}"
        )


    elif type == 'coefs':
        poly_order = kwargs['poly_order']

        f_ident = (
            f"order{poly_order}_"
            f"c{config['c']}_"
            f"dt{config['dt']}_"
            f"si{config['si']}_"
            f"train{config['train_perc']*config['t_total']}_"
            f"rs{config['seed']}"
        )

    elif type == 'ar1':

        f_ident = (
            f"c{config['c']}_"
            f"dt{config['dt']}_"
            f"si{config['si']}_"
            f"train{config['train_perc']*config['t_total']}_"
            f"rs{config['seed']}"
        )

    else: 
        raise ValueError(f"Unknown type '{type}'. Cannot find identifier string")
    return f_ident
