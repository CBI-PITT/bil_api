

#https://cloud.google.com/appengine/docs/flexible/integrating-with-analytics?tab=python

#https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters

# Environment variables are defined in app.yaml.
GA_TRACKING_ID = 'G-ZX9TCXH925'


def track_event(category, action, label=None, value=0):
    """
    Send an event to Google Analytics using the Measurement Protocol.

    Args:
        category (str): Event category (e.g., "UX", "Button").
        action (str): Event action (e.g., "click", "submit").
        label (str, optional): Event label (e.g., "Submit Button"). Defaults to None.
        value (int, optional): Event value (must be an integer). Defaults to 0.

    Additional Parameters Sent:
        - 'ds': Data source, set to "brainpi".
        - 'uid': User ID (if available).
        - 'sr': Screen resolution.
        - 'vp': Viewer port size.
        - 'uip': IP override.
        - 'dl': Document location (URL).
        - 'cid': Anonymous Client ID (set to "555" for simplicity).
        - 'ua': User agent string.

    Raises:
        requests.exceptions.RequestException: If the HTTP request to Google Analytics fails.
    """
    data = {
        'v': '1',  # API Version.
        'tid': GA_TRACKING_ID,  # Tracking ID / Property ID.
        't': 'event',  # Event hit type. **REQUIRED**
        'ec': category,  # Event category. **REQUIRED** https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#ec
        'ea': action,  # Event action. **REQUIRED** https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#ea
        'el': label,  # Event label. https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#el
        'ev': value, # Event value, must be an integer https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#ev


        'ds': 'brainpi', # Datasource https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#ds
        'uid': userid, # https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#uid
        'sr': screen_resolution, # https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#sr
        'vp': viewer_port_size, # https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#vp
        'uip': ip_override, # https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#uip
        'dl': document_location, # https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#dl
        'cid': '555', # https://developers.google.com/analytics/devguides/collection/protocol/v1/parameters#cid




        'ua': 'Opera/9.80 (Windows NT 6.0) Presto/2.12.388 Version/12.14'
    }

    response = requests.post(
        'https://www.google-analytics.com/collect', data=data)

    # If the request fails, this will raise a RequestException. Depending
    # on your application's needs, this may be a non-error and can be caught
    # by the caller.
    response.raise_for_status()


# v=1                                   // Version.
# &tid=UA-XXXXX-Y                       // Tracking ID / Property ID.
# &cid=555                              // Anonymous Client ID.
# &t=event                              // Event hit type.
# &ec=UX                                // Event Category. Required.
# &ea=click                             // Event Action. Required.
# &el=Results                           // Event label.
#
# &pa=click                             // Product action (click). Required.
# &pal=Search%20Results                 // Product Action List.
# &pr1id=P12345                         // Product 1 ID. Either ID or name must be set.
# &pr1nm=Android%20Warhol%20T-Shirt     // Product 1 name. Either ID or name must be set.
# &pr1ca=Apparel                        // Product 1 category.
# &pr1br=Google                         // Product 1 brand.
# &pr1va=Black                          // Product 1 variant.
# &pr1ps=1                              // Product 1 position.

