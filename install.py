import subprocess
"""
Use this to install all requirements library
"""
def install_requirements():
    subprocess.call(['pip', 'install', '-r', 'requirements.txt'])


## Main function.    
if __name__ == '__main__':
    install_requirements()
    # Rest of your code goes here
