import re
from urllib.parse import urlparse


class URLFeatureExtractor:

    def __init__(self):
        self.feature_names = [
            "having_IP_Address",
            "URL_Length",
            "Shortining_Service",
            "having_At_Symbol",
            "double_slash_redirecting",
            "Prefix_Suffix",
            "having_Sub_Domain",
            "port",
            "HTTPS_token"
        ]

    def extract_features(self, url: str):

        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""

        # 1. IP Address
        # Dataset encoding:
        # -1 = phishing/suspicious
        #  1 = legitimate
        ip_pattern = (
            r"^(?:\d{1,3}\.){3}\d{1,3}$"
        )

        if re.match(ip_pattern, hostname):
            having_ip_address = -1
        else:
            having_ip_address = 1

        # 2. URL Length
        if len(url) < 54:
            url_length = 1
        elif len(url) <= 75:
            url_length = 0
        else:
            url_length = -1

        # 3. URL Shortening Service
        shortening_services = [
            "bit.ly",
            "goo.gl",
            "shorte.st",
            "go2l.ink",
            "x.co",
            "ow.ly",
            "t.co",
            "tinyurl.com",
            "tr.im",
            "is.gd",
            "cli.gs",
            "yfrog.com",
            "migre.me",
            "tiny.cc",
            "bit.do",
            "adf.ly",
            "bitly.com",
            "cutt.ly",
            "lnkd.in"
        ]

        if any(service in hostname.lower() for service in shortening_services):
            shortening_service = -1
        else:
            shortening_service = 1

        # 4. @ Symbol
        if "@" in url:
            having_at_symbol = -1
        else:
            having_at_symbol = 1

        # 5. Double slash redirecting
        path = parsed_url.path

        if "//" in path:
            double_slash_redirecting = -1
        else:
            double_slash_redirecting = 1

        # 6. Prefix / Suffix
        if "-" in hostname:
            prefix_suffix = -1
        else:
            prefix_suffix = 1

        # 7. Subdomain
        dot_count = hostname.count(".")

        if dot_count <= 2:
            having_sub_domain = 1
        elif dot_count == 3:
            having_sub_domain = 0
        else:
            having_sub_domain = -1

        # 8. Port
        if parsed_url.port is None:
            port = 1
        elif parsed_url.port in [80, 443]:
            port = 1
        else:
            port = -1

        # 9. HTTPS token inside domain
        domain_without_scheme = hostname.lower()

        if "https" in domain_without_scheme or "http" in domain_without_scheme:
            https_token = -1
        else:
            https_token = 1

        return {
            "having_IP_Address": having_ip_address,
            "URL_Length": url_length,
            "Shortining_Service": shortening_service,
            "having_At_Symbol": having_at_symbol,
            "double_slash_redirecting": double_slash_redirecting,
            "Prefix_Suffix": prefix_suffix,
            "having_Sub_Domain": having_sub_domain,
            "port": port,
            "HTTPS_token": https_token
        }