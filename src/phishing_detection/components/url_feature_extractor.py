import math
import re
from urllib.parse import urlparse


class URLFeatureExtractor:

    def __init__(self):
        self.feature_names = [
            "url_len",
            "dom_len",
            "is_ip",
            "tld_len",
            "subdom_cnt",
            "letter_cnt",
            "digit_cnt",
            "special_cnt",
            "eq_cnt",
            "qm_cnt",
            "amp_cnt",
            "dot_cnt",
            "dash_cnt",
            "under_cnt",
            "letter_ratio",
            "digit_ratio",
            "spec_ratio",
            "is_https",
            "slash_cnt",
            "entropy",
            "path_len",
            "query_len"
        ]

    def calculate_entropy(self, value):
        if not value:
            return 0.0

        frequency = {}

        for char in value:
            frequency[char] = frequency.get(char, 0) + 1

        length = len(value)
        entropy = 0.0

        for count in frequency.values():
            probability = count / length
            entropy -= probability * math.log2(probability)

        return entropy

    def extract_features(self, url: str):

        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL must be a non-empty string.")

        url = url.strip()

        # Add scheme temporarily when parsing URLs such as www.google.com
        parsed_url = urlparse(
            url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url)
            else "http://" + url
        )

        hostname = parsed_url.hostname or ""

        domain_parts = hostname.split(".")

        if len(domain_parts) >= 2:
            domain = ".".join(domain_parts[-2:])
        else:
            domain = hostname

        dom_len = len(domain)

        # -----------------------------
        # URL length
        # -----------------------------
        url_len = len(url)

        # -----------------------------
        # Domain
        # -----------------------------
        dom_len = len(domain)

        # -----------------------------
        # IP detection
        # -----------------------------
        is_ip = 0

        if re.match(
            r"^(?:\d{1,3}\.){3}\d{1,3}$",
            domain
        ):
            is_ip = 1

        # -----------------------------
        # TLD
        # -----------------------------
        tld = ""

        if domain and "." in domain:
            tld = domain.split(".")[-1]

        tld_len = len(tld)

        # -----------------------------
        # Subdomain count
        # -----------------------------
        subdom_cnt = max(len(domain_parts) - 2, 0)

        # -----------------------------
        # Character counts
        # -----------------------------
        letter_cnt = sum(char.isalpha() for char in url)

        digit_cnt = sum(char.isdigit() for char in url)

        special_cnt = sum(
            not char.isalnum()
            for char in url
        )

        eq_cnt = url.count("=")
        qm_cnt = url.count("?")
        amp_cnt = url.count("&")
        dot_cnt = url.count(".")
        dash_cnt = url.count("-")
        under_cnt = url.count("_")
        slash_cnt = url.count("/")

        # -----------------------------
        # Ratios
        # -----------------------------
        if url_len > 0:
            letter_ratio = letter_cnt / url_len
            digit_ratio = digit_cnt / url_len
            spec_ratio = special_cnt / url_len
        else:
            letter_ratio = 0.0
            digit_ratio = 0.0
            spec_ratio = 0.0

        # -----------------------------
        # HTTPS
        # -----------------------------
        is_https = int(parsed_url.scheme.lower() == "https")

        # -----------------------------
        # Entropy
        # -----------------------------
        entropy = self.calculate_entropy(url)

        # -----------------------------
        # Path / Query
        # -----------------------------
        path = parsed_url.path or ""
        query = parsed_url.query or ""

        path_len = len(path)
        query_len = len(query)

        # -----------------------------
        # Final feature dictionary
        # -----------------------------
        features = {
            "url_len": url_len,
            "dom_len": dom_len,
            "is_ip": is_ip,
            "tld_len": tld_len,
            "subdom_cnt": subdom_cnt,
            "letter_cnt": letter_cnt,
            "digit_cnt": digit_cnt,
            "special_cnt": special_cnt,
            "eq_cnt": eq_cnt,
            "qm_cnt": qm_cnt,
            "amp_cnt": amp_cnt,
            "dot_cnt": dot_cnt,
            "dash_cnt": dash_cnt,
            "under_cnt": under_cnt,
            "letter_ratio": letter_ratio,
            "digit_ratio": digit_ratio,
            "spec_ratio": spec_ratio,
            "is_https": is_https,
            "slash_cnt": slash_cnt,
            "entropy": entropy,
            "path_len": path_len,
            "query_len": query_len
        }

        return features