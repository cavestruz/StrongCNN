import ConfigParser

def parse_configfile(cfgdir) :
    '''
    |
    |   Return a nested dictionary for parameter grid and pipeline
    |
    '''

    import os
    
    assert(os.path.exists(cfgdir))

    cfg_dict = {}
    Config = ConfigParser.ConfigParser()
    Config.optionxform = str # Otherwise, options are lowercased
    Config.read(cfgdir+'/config.ini')
    for section in Config.sections() : 
        cfg_dict[section] = { option: Config.get(section, option) \
                                  for option in Config.options(section) }

    return cfg_dict        
