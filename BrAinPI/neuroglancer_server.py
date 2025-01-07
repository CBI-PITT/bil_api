import neuroglancer
import urllib.request
import signal
from logger_tools import logger
# Project specific imports
import config_tools
def get_server():
    """
    Lauch the neuroglancer server
    """
    settings = config_tools.get_config('settings.ini')
    ip = settings.get('neuroglancer','local_ip')
    port = settings.getint('neuroglancer','local_port')
    token = 'base'

    # When Multiple processes are running, we only want 1 instance of the neuroglancer server
    # This will test for whether the server is running and if:
    #   False: Return server
    #   True: Return None
    code = None
    try:
        ng_server_url = f'http://{ip}:{port}/v/{token}/'
        with urllib.request.urlopen(ng_server_url) as r:
            code = r.getcode()
    except:
        pass

    if code == 200:
        return
    else:
        try:
            neuroglancer.set_server_bind_address(bind_address=ip, bind_port=port)
            viewer = neuroglancer.UnsynchronizedViewer(token='base')
            # viewer = neuroglancer.Viewer()
            logger.success(viewer)
            return viewer
        except OSError:
            del viewer
            return
        except:
            del viewer
            return

def keep_alive():
    """
    Wait for signals (Ctrl+C will exit)
    """
    signal.pause() 

if __name__ == '__main__':
        viewer = get_server()
        if viewer:
            keep_alive()