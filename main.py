
from colorama import Fore, Back, Style
import pyfiglet
import bootstrap
import api

APP_NAME = 'MuseQuill.Ink'

def app_banner(app_name:str):
    """Different font styles"""    
    text = pyfiglet.figlet_format(app_name, font='slant')
    print(Fore.GREEN + text + Style.RESET_ALL)

def main():
    api.run()

if __name__ == "__main__":
    app_banner(APP_NAME)
    main()
