# -*- coding: utf-8 -*-
import cStringIO
import cgi
import os
import traceback

from urlparse import urlparse, parse_qs
import requests
import tornado.httpserver
import tornado.ioloop
import tornado.web
import numpy as np
import urllib
from utils import *
from handler_base import *

import cv2

SERVER_PORT = 4443  # http:80, https:443

def getFormValue(f):
    if f.file is None:
        return f.value
    else:
        return f.file.read()


class S(HandlerBase):
    @tornado.gen.coroutine
    def get(self):
        url_path = self.request.uri[1:]

        o = urlparse(url_path)
        q = parse_qs(o.query)

        try:
            fname = o.path.split('../')[0]
            ext = fname.split('.')[-1].lower()
            self.set_headers(FILE_EXTS[ext])
            self.write(open('static/' + fname, 'rb').read())
            return
        except:
            pass
    @tornado.gen.coroutine
    def post(self):
        url_path = self.request.uri[1:]
        o = urlparse(url_path)
        q = parse_qs(o.query)

        postvars = cgi.FieldStorage(
            fp=cStringIO.StringIO(self.request.body),
            headers=self.request.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.request.headers['Content-Type']})

        

        if o.path == 'photo':
            link = getFormValue(postvars['link'])
            photo = getFormValue(postvars['photo'])
            file_name_from_page = "photos/photo_from_page.jpg"
            file_name_from_user = "photos/photo_from_user.jpg"
            open(file_name_from_user, 'wb').write(photo)
            urllib.urlretrieve(link, file_name_from_page)
            im_from_page = cv2.imread(file_name_from_page)
            im_from_user = cv2.imread(file_name_from_user)
            payloadz = {'img_from':im_from_user.tolist(), 'img_to':im_from_page.tolist()}
            dataw = json.dumps(payloadz, default=str)
            output = requests.get('http://13.90.200.154:9213/swap',data=dataw)
            # output = requests.get('http://13.90.200.154:9213/swap', data={'text': 'qwerty', 'img_from': im_from_user.dumps(), 'img_to': im_from_page.dumps()})
            cv2.imwrite("static/" + file_name_from_user, np.asarray(output.json(), dtype=np.uint8))
            self.write(file_name_from_user)
            self.set_status(200)
            return

        self.set_status(404)


        if o.path == 'video':
            video = getFormValue(postvars['video'])
            name =  getFormValue(postvars['name'])
            open('videos/%s.jpg' % name, 'wb').write(video)
            self.set_status(200)
            return


if __name__ == "__main__":
    try:
        logging.debug('Starting httpd on %s...' % SERVER_PORT)

        settings = {'compress_response': True,
                    "cookie_secret": 'L8LwECiNRxq2N0N2eGxx9MZlrpmuMEimlydNX/vt1LM=',
                    "xheaders": True}
        ssl_options = {
            "certfile": os.path.expanduser('~/certs/meas_tardis3d_ru.crt'),
            "keyfile": os.path.expanduser('~/certs/meas_tardis3d_ru.key'),
        }

        app = tornado.web.Application([
            (r"/.*", S)
        ], **settings)
        
        srv = tornado.httpserver.HTTPServer(app, ssl_options=ssl_options)
        srv.listen(SERVER_PORT)

        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        pass
    except:
        logging.debug(traceback.format_exc())

    tornado.ioloop.IOLoop.current().stop()
    logging.debug("Exiting")
