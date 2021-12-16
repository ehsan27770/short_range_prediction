import hashlib
import urllib
import time

#'https://coronavax.unisante.ch/'

local_data = urllib.request.urlopen('https://www.google.com/finance/').read()
local_hash = hashlib.md5(remote_data).hexdigest()


for i in range(50):
    remote_data = urllib.request.urlopen('https://www.google.com/finance/').read()
    remote_hash = hashlib.md5(remote_data).hexdigest()
    if remote_hash == local_hash:
        print('no changed')
    else:
        print('warning')
    local_hash = remote_hash
    time.sleep(1)
