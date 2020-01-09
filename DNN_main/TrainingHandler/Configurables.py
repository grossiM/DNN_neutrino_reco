import sys

def checkForProperty(config, section, option):
    if not config.has_option(section, option):
        raise ValueError('Error: Property '+option+' is required in the section '+section)

def validate(config):
    checkForProperty(config,'output','output-folder')
    checkForProperty(config,'training','epochs')
    checkForProperty(config,'training','batch-size')
    checkForProperty(config,'training','model')
    checkForProperty(config,'training','discriminating-variable')
    checkForProperty(config,'input','data-train')
    checkForProperty(config,'input','data-val')
    