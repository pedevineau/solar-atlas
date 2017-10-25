import email
import imaplib
from datetime import datetime
import re
from general_utils import daytimeconv

__author__ = 'marek'


class EmailMessage():
    """Wraps email message."""

    def __init__(self, uid, message_string):
        msg = email.message_from_string(message_string)
        self.uid = uid
        self.sent_from = msg['From']
        self.sent_to = msg['To']
        self.subject = msg['Subject']
        # Unique message ID:
        self.message_id = msg.get('Message-ID')
        # datetime is timezone naive, internal date of message on server, consider TZ of mail server
        # message date example: Thu, 12 Jun 2014 00:01:38 +0200
        # cut off timezone
        pattern = r'^(.*) \+.*$'
        re_match = re.search(pattern, msg['Date'], re.MULTILINE)
        datetime_str_witthout_timezone = re_match.group(1)
        dt = datetime.strptime(datetime_str_witthout_timezone, "%a, %d %b %Y %H:%M:%S")
        self.date = dt
        # all headers:
        parser = email.parser.HeaderParser()
        self.headers = parser.parsestr(msg.as_string())
        # body
        maintype = msg.get_content_maintype()
        if maintype == 'text':
            self.body = msg.get_payload(decode=True)  # decode=True to preserve utf-8
        else:
            # raise Exception("Multipart email message is not supported.")
            for part in msg.walk():
                # print part
                if part.get_content_type() == "text/plain":
                    self.body = part.get_payload(decode=True)
                else:
                    continue


    def __repr__(self):
        return 'Email UID: %s, %s, %s, SENT-FROM: %s, ID: %s' % (self.uid, self.date.strftime("%Y-%m-%d %H:%M"),
                                                                 self.subject, self.sent_from, self.message_id)


class EmailChecker():
    """meaning of verbose:
        Default: 0
        1 = little output
        5 = very talkative"""
    def __init__(self, host, login, password, mailbox, verbose=0):
        imaplib.Debug = verbose
        # Connect:
        imap = imaplib.IMAP4_SSL(host)
        imap.login(login, password)
        # List mailbox names:
        # imap.list()
        #Select a mailbox (or a label in case of Gmail). Returned data is the count of messages in mailbox:
        imap.select(mailbox)
        self.imap = imap
        self.mailbox = mailbox

    def set_flag(self, msg_uids_str, flag):
        """msg_uids_str example '1234 2354 2343
        Store flag to one or more messages on server."""
        self.imap.store(msg_uids_str, '+FLAGS', '%s' % flag)

    def remove_flag(self, msg_uids_str, flag):
        """msg_uids_str example '1234 2354 2343
        Deletes flag from one or more messages on server."""
        self.imap.store(msg_uids_str, '-FLAGS', '%s' % flag)

    def get_flags(self, msg_uid_str):
        """msg_uid_str example '1234', note only one uid!
        the raw output from fetch can be:
        ('OK', ['1729 (FLAGS (PROCESSED \\Seen))'])"""
        status, response = self.imap.fetch(msg_uid_str, '(FLAGS)')
        pattern = r'^.*\(FLAGS \((.*)\).*$'
        # print response
        re_match = re.search(pattern, response[0], re.MULTILINE)
        flags_str = re_match.group(1)
        flag_list = flags_str.split()
        return flag_list

    def mailbox_status(self):
        status, response = self.imap.status(self.mailbox, '(MESSAGES)')
        return response

    def list_by_filter(self, since_date_yyymmdd, before_date_yyymmdd=None, subject=None, has_flag=None, has_not_flag=None):
        """by subject: matches all mails where 'subject' occurs anywhere in SUBJECT
           by keyword: This field in email msg. contains keywords or phrases, separated  by commas.
           More here: http://tools.ietf.org/html/rfc822
        """
        since_date = daytimeconv.yyyymmdd2date(since_date_yyymmdd)
        before_date = None
        if before_date_yyymmdd:
            before_date = daytimeconv.yyyymmdd2date(before_date_yyymmdd)
        criterion = '(SINCE "%s"' % since_date.strftime("%d-%b-%Y")
        if before_date:
            criterion += ' BEFORE "%s"' % before_date.strftime("%d-%b-%Y")
        if subject:
            criterion += " SUBJECT '%s'" % subject
        if has_flag:
            criterion += ' KEYWORD "%s"' % has_flag
        if has_not_flag:
            criterion += ' NOT KEYWORD "%s"' % has_not_flag
        criterion += ")"
        # print criterion
        search_status, response = self.imap.uid('search', None, criterion)
        messages_found = []
        if response:
            uid_list = response[0].split(' ')
            for uid in uid_list:
                if not uid:  # UID is empty string if nothing is found
                    continue
                email_message = self.get_by_uid(uid)
                messages_found.append(email_message)
        return messages_found

    def get_by_uid(self, uid_str):
        if not uid_str:
            raise Exception("Empty message UID was passed, cannot continue.")
        search_status, response = self.imap.uid('fetch', uid_str, '(RFC822)')
        msg_str = response[0][1]
        email_message = EmailMessage(uid_str, msg_str)
        return email_message

    def list_newer_mails(self, last_uid):
        """Note: messages are in order they were added to the mailbox, which is not necessarily by date!"""
        criterion = '(UID %s:*)' % last_uid
        search_status, response = self.imap.uid('search', None, criterion)
        msg_uids = response[0].split()
        messages_found = []
        if msg_uids:
            for uid in msg_uids:
                email_message = self.get_by_uid(uid)
                messages_found.append(email_message)
        return messages_found


if __name__ == "__main__":
    checker = EmailChecker("imap.gmail.com", "techsolargis@geomodel.eu", "kgdwrblzrezlahml", "climdata-delivery-request", verbose=0)
    print checker
    # checker = EmailChecker("imap.gmail.com", "tech@geomodel.eu", "hikwjloljvyhorhe", "climdata-delivery-request", verbose=0)
    # mail = checker.get_by_uid('6251')
    # print mail
    # print mail.body
    # print checker.mailbox_status()
    # mails = checker.list_newer_mails('1735')
    # list by filter in gmail ui: "after:2014/06/11 before:2014/06/12 from:no-reply@solargis.info subject:"solargis resource request""
    # mails = checker.list_by_filter('20140611', before_date_yyymmdd='20140611', subject="SG-5508-1406-13-1", flag=None)
    # mails = checker.list_by_filter('20140611', before_date_yyymmdd='20140612', subject="SG-5508-1406-13-1", flag='PROCESSED')
    mails = checker.list_by_filter(since_date_yyymmdd='20160407', has_not_flag="CLIMDATA_SEEN")
    # mails = checker.list_by_filter(since_date_yyymmdd='20160407')
    print len(mails)
    for mail in mails:
        print mail.body[:50]
        print checker.get_flags(mail.uid)
        # checker.remove_flag(mail.uid, 'PROCESSED')
        # print checker.get_flags(mail.uid)
        # checker.set_flag(mail.uid,'CLIMDATA_SEEN')