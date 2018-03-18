# -*- coding: utf-8 -*-
import traceback
from time import strftime, time, localtime
from urlparse import urlparse, parse_qs
import urllib
from mimetypes import guess_type
import tornado.web


from utils import *


HTML = 'text/html'
CSS = 'text/css'
ICO = 'image/ico'
GIF = 'image/gif'
JPG = 'image/jpeg'
PNG = 'image/png'
SVG = 'image/svg+xml'
JS = 'application/x-javascript'
TXT = 'text/plain'
JSON = 'application/json'
ZIP = 'application/zip'
FILE_EXTS = {'html': HTML, 'css': CSS, 'ico': ICO, 'gif': GIF, 'jpg': JPG, 'jpeg': JPG, 'png': PNG, 'js': JS,
             'txt': TXT, 'json': JSON, 'svg': SVG}


class HandlerBase(tornado.web.RequestHandler):
    def prepare(self):
        """Redirects to using https scheme."""
        if self.request.protocol == 'http':
            self.redirect('https://' + self.request.host, permanent=False)
    
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Server", "Tardis3DWebServer/1.0")
        

    def set_headers(self, t='text/html'):
        self.set_status(200)
        self.set_header('Content-type', t)

    def get_current_user(self):
        secure_cookie = self.get_secure_cookie("user")
        if secure_cookie is None:
            secure_cookie = ""

            url_path = self.request.uri[1:]
            o = urlparse(url_path)
            q = parse_qs(o.query)

            if 'token' in q:
                data = decodeToken(q['token'][0])
                if data is not None:
                    login, ts = data  # parse token data
                    if ts > time():
                        return login

        user = decodeToken(secure_cookie)
        if type(user) is list and user[1] > time():
            self.clear_cookie("user")
            self.set_secure_cookie("user", genToken(user[0]))
            return user[0]
        
        return None

    def set_current_user(self, token):
        if token:
            self.set_secure_cookie("user", token)
        else:
            self.clear_cookie("user")
    
#    def redirect_to_login(self):
#        url_path = self.request.uri[1:]
#        self.redirect("/login?next="+self.request.uri[1:])

class SimplestFileHandler(HandlerBase):
    def initialize(self, path):
        self.root = os.path.abspath(path) + os.path.sep

    def get(self, path):
        if os.path.sep != "/":
            path = path.replace("/", os.path.sep)
        abspath = os.path.join(self.root, path)
        
        if not os.path.exists(abspath):
            self.set_status(404)
            return
        
        mime_type, encoding = guess_type(abspath)
        if mime_type:
            self.set_headers(mime_type)
        
        file = open(abspath, "rb")
        try:
            self.write(file.read())
        finally:
            file.close()


#class ScanDataFileHandler(HandlerBase):
#    def openScanHistory(self, data_fn):
#        """ Assembles history into scan_data """
#        try:
#            history = json.load(open(data_fn, 'rt'))
#        except:
#            history = []
#    
#        # merge history
#        data = {}
#        for d in history:
#            data.update(d)
#        return history, data
#    
#    def initialize(self, path):
#        self.root = os.path.abspath(path) + os.path.sep
#
#    def get(self, path):
#        if os.path.sep != "/":
#            path = path.replace("/", os.path.sep)
#        abspath = os.path.join(self.root, path)
#        
#        if not os.path.exists(abspath):
#            self.set_status(404)
#            return
#        
#        db = DBConnecting()
#        h,d = openScanHistory(abspath)
#        d['erp_data'].update(json.loads(db.get_item()))
#        self.write(file.read())
