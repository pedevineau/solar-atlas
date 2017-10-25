'''
Created on Mar 4, 2011
@author: tomas
'''

import sys
import os
import smtplib
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders
import mimetypes

def isiterable(obj):
    return hasattr(obj, '__iter__')

def mail(serverURL=None, sender='', to='', subject='', text=''):
    """
    Usage:
    mail('somemailserver.com', 'me@example.com', 'someone@example.com', 'test', 'This is a test')
    """
    headers = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (sender, to, subject)
    message = headers + text
    mailServer = smtplib.SMTP(serverURL)
    mailServer.sendmail(sender, to, message)
    mailServer.quit()

def addSenderToSubject(sender, subject):
    result = sender + ': ' + subject
    if len(sender.split('@')) > 1:
        result = sender.split('@')[0]+ ': ' + subject
    return result


#it is here only for back compatibility 
def mail_ssl(serverURL=None, serverPort=465, serverUser='', serverPassword='', sender='', to='', subject='', text=''):
    """
    Usage:
    mail('somemailserver.com', 'me@example.com', 'someone@example.com', 'test', 'This is a test')
    """
    
    if sys.version_info < (2, 6):
        import ssmtplib
        mailServer = ssmtplib.SMTP_SSL(serverURL,serverPort)
        headers = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (sender, to, subject)
    else:
        mailServer = smtplib.SMTP_SSL(serverURL,serverPort)
        headers = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (sender, to, addSenderToSubject(sender, subject))
    # syntax error in 2.4, ok in 2.5
    message = headers + text
#    message = text
    
    mailServer.ehlo()
    mailServer.login(serverUser,serverPassword)
#    mailServer.ehlo()
    mailServer.sendmail(sender, to, message)
    mailServer.quit()


def _add_attachment(out_msg, f):
    '''
    adapted from
    http://ginstrom.com/scribbles/2009/03/15/a-module-to-send-email-simply-in-python/
    '''
#    print 'A'
#    print type(f)
#    import posixpath
#    print posixpath.splitext(f)
#    print 'B'
    ctype, encoding = mimetypes.guess_type(f)
#    print 'C'
#    print ctype, encoding
#    exit()
    if ctype is None or encoding is not None:
        # No guess could be made, or the file is encoded (compressed), so
        # use a generic bag-of-bits type.
        ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)
    if maintype == 'text':
        # Note: we should handle calculating the charset
        msg = MIMEText(open(f,"rb").read(), _subtype=subtype)
    elif maintype == 'image':
        msg = MIMEImage(open(f,"rb").read(), _subtype=subtype)
    elif maintype == 'audio':
        msg = MIMEAudio(open(f,"rb").read(), _subtype=subtype)
    else:
        msg = MIMEBase(maintype, subtype)
        msg.set_payload(open(f,"rb").read())
        # Encode the payload using Base64
        Encoders.encode_base64(msg)
    # Set the filename parameter
    msg.add_header('Content-Disposition',
            'attachment',
            filename=os.path.basename(f))
    out_msg.attach(msg)



def mail_ssl2(serverURL=None, serverPort=465, serverUser='', serverPassword='', sender='', to='', subject='', text='', attachments=[]):
    """
    Usage:
    mail('somemailserver.com', 'me@example.com', 'someone@example.com', 'test', 'This is a test')
    """
    
    if sys.version_info < (2, 6):
        import ssmtplib
        mailServer = ssmtplib.SMTP_SSL(serverURL,serverPort)
    else:
        mailServer = smtplib.SMTP_SSL(serverURL,serverPort)
        subject=addSenderToSubject(sender, subject)

#    headers = "From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n" % (sender, to, subject)
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = COMMASPACE.join(to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
    
    if (text is not None) and (text != ''):
        msg.attach( MIMEText(text) )
    

    for f in attachments:
        _add_attachment(msg, f)
    
    msg = msg.as_string()
    
    
    mailServer.ehlo()
    mailServer.login(serverUser,serverPassword)
#    mailServer.ehlo()
    mailServer.sendmail(sender, to, msg)
    mailServer.quit()


def mail_process_message_ssl(sender_from='', reciever_to='', message='', subject='', serverURL='smtp.googlemail.com', port=465, user='tech@solargis.com', password='elbsjdhycuwmgdec',attachments=[]):
    #mail processing message wrapper
    
    #make sender from computer (host) name
    if (sender_from is None) or (sender_from==''):
        import socket
        hostname = socket.gethostname()
        sender_from=hostname+'@solargis.com'

    #make subject from python file script name
    if (subject is None) or (subject==''):
        subject=sys.argv[0]
        if subject.find('/')  >=0:
            subject=subject[subject.rindex('/')+1:]
        if subject.find('.') >=0:
            subject=subject[:subject.rindex('.')]
    
    #finaly send mail
    if not isiterable(reciever_to):
        recievers=[reciever_to]
    else: 
        recievers=reciever_to
    try:
        mail_ssl2(serverURL=serverURL, serverPort=port, serverUser=user, serverPassword=password, sender=sender_from, to=recievers, subject=subject, text=message, attachments=attachments)
    except:
        print sys.exc_info()
        print "problem sending mail"


if __name__ == '__main__':
    ###test: send simple mail
    # $ python src/general_utils/basic_mail.py -t tomas.cebecauer@solargis.com -m "Hi, I'm just testing shortest possible way"

    ###test: send email with attachment
    # $ python src/general_utils/basic_mail.py -t tomas.cebecauer@solargis.com -a /tmp/GHI_20120325_20120325.png -m 'This is test of explicit definition of mail parameters' -s 'my subject' -u tech@solargis.com -p elbsjdhycuwmgdec --smtp smtp.googlemail.com --port 465

    ###test: send mail to multireceiver
    # $ python src/general_utils/basic_mail.py -t tomas.cebecauer@solargis.com marek.caltik@solargis.com cebecauer@solargis.com -m 'This is test of sending mail to multiple receivers defined as list' -s 'my subject' -u tech@solargis.com -p elbsjdhycuwmgdec --smtp smtp.googlemail.com --port 465

    import argparse
    parser=argparse.ArgumentParser(description='Sending emails via google account.')
    parser.add_argument('-s', '--subject', dest='subject', help='Subject (default: \'Testing email configuration\')', default="Testing email configuration")
    parser.add_argument('-a', '--attachment', dest='attachments', nargs='+', help='Attachment files')
    parser.add_argument('-t', '--to', dest='reciever_to', nargs='+', help='Receiver(s) of sending email', required=True)
    parser.add_argument('-f', '--from', dest='sender_from', help='Send email from specifig address (default: {hostname}@solargis.com)')
    parser.add_argument('-u', '--user', dest='user', help='Google login primary email (default: tech@solargis.com)')
    parser.add_argument('-p', '--password', dest='password', help='Google account or application password (default exists)')
    parser.add_argument('-m', '--message', dest='message', help='Body fo sending message (default is empty)')
    parser.add_argument('--smtp', dest='serverURL', help='Name of SMTP server (default: smtp.googlemail.com)')
    parser.add_argument('--port', dest='port', type=int, help='Port of SMTP server (default: 465)')
    args=vars(parser.parse_args())
    for k in args.keys():
        if args[k] == None:
            del args[k]
    mail_process_message_ssl(**args)

