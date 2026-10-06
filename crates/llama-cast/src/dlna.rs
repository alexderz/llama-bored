//! The UPnP MediaServer documents and ContentDirectory answers.
//!
//! One container (`0`) holding one live video item (`1`). The item is
//! MPEG-TS advertised as `video/mpeg` with the `MPEG_TS_HD_NA_ISO` profile:
//! that is the variant Roku Media Player (TCL Roku TVs) listed and played;
//! with `video/mp2t` it reported "No compatible videos found".
//!
//! Requests are parsed for exactly two values, `ObjectID` and
//! `BrowseFlag`; the rest of a SOAP body (Filter, StartingIndex, sort) is
//! ignored. Nothing here does I/O.

/// Every route the HTTP server answers.
pub const DESC_PATH: &str = "/desc.xml";
pub const CDS_SCPD_PATH: &str = "/cds.xml";
pub const CMS_SCPD_PATH: &str = "/cms.xml";
pub const CDS_CONTROL_PATH: &str = "/ctl/cds";
pub const CMS_CONTROL_PATH: &str = "/ctl/cms";
pub const CDS_EVENT_PATH: &str = "/evt/cds";
pub const CMS_EVENT_PATH: &str = "/evt/cms";
pub const STREAM_PATH: &str = "/live.ts";

pub const DEVICE_TYPE: &str = "urn:schemas-upnp-org:device:MediaServer:1";
pub const CDS_TYPE: &str = "urn:schemas-upnp-org:service:ContentDirectory:1";
pub const CMS_TYPE: &str = "urn:schemas-upnp-org:service:ConnectionManager:1";

/// The stream's MIME type.
pub const STREAM_MIME: &str = "video/mpeg";
/// `DLNA.ORG_FLAGS` primary bits (the first 8 of its 32 hex digits) set
/// for this live stream: streaming transfer mode (tm-s), background
/// transfer (tm-b), HTTP connection stalling, DLNA 1.5. Not set: sender
/// paced, limited-range seek (lop-npt, lop-bytes) and s0/sN-increasing; with
/// `DLNA.ORG_OP=00` (no time or byte seek) a player treats it as a stream
/// to play from where it joins. This is the set the Roku accepted.
pub const FLAG_STREAMING: u32 = 1 << 24;
pub const FLAG_BACKGROUND: u32 = 1 << 22;
pub const FLAG_HTTP_STALLING: u32 = 1 << 21;
pub const FLAG_DLNA_V15: u32 = 1 << 20;
/// The DLNA fourth field of `protocolInfo`.
pub const DLNA_FEATURES: &str =
    "DLNA.ORG_PN=MPEG_TS_HD_NA_ISO;DLNA.ORG_OP=00;DLNA.ORG_FLAGS=01700000000000000000000000000000";
/// The exact `protocolInfo` the Roku accepted.
pub const PROTOCOL_INFO: &str = "http-get:*:video/mpeg:DLNA.ORG_PN=MPEG_TS_HD_NA_ISO;DLNA.ORG_OP=00;DLNA.ORG_FLAGS=01700000000000000000000000000000";
/// Advertised resolution (the frame size).
pub const RESOLUTION: &str = "1920x1080";
/// Content-Type of every XML answer.
pub const XML_CONTENT_TYPE: &str = "text/xml; charset=\"utf-8\"";

/// The root container and the one item.
pub const ROOT_ID: &str = "0";
pub const ITEM_ID: &str = "1";

/// What the documents need to know about this server.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Device {
    /// Friendly name (already validated: printable, 1..=64 chars).
    pub name: String,
    /// The video item's title (likewise).
    pub title: String,
    /// `uuid:...`.
    pub udn: String,
    /// `http://ADDR:PORT`.
    pub base_url: String,
}

impl Device {
    /// The stream URL.
    #[must_use]
    pub fn stream_url(&self) -> String {
        format!("{}{STREAM_PATH}", self.base_url)
    }
}

/// XML text escape.
#[must_use]
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

/// `/desc.xml`.
#[must_use]
pub fn description(dev: &Device) -> String {
    format!(
        concat!(
            "<?xml version=\"1.0\"?>\n",
            "<root xmlns=\"urn:schemas-upnp-org:device-1-0\">",
            "<specVersion><major>1</major><minor>0</minor></specVersion>\n",
            "<device><deviceType>{dt}</deviceType><friendlyName>{name}</friendlyName>\n",
            "<manufacturer>llama-bored</manufacturer><modelName>llama-bored cast</modelName>",
            "<UDN>{udn}</UDN>\n",
            "<serviceList>\n",
            "<service><serviceType>{cds}</serviceType>",
            "<serviceId>urn:upnp-org:serviceId:ContentDirectory</serviceId>",
            "<SCPDURL>{cds_scpd}</SCPDURL><controlURL>{cds_ctl}</controlURL>",
            "<eventSubURL>{cds_evt}</eventSubURL></service>\n",
            "<service><serviceType>{cms}</serviceType>",
            "<serviceId>urn:upnp-org:serviceId:ConnectionManager</serviceId>",
            "<SCPDURL>{cms_scpd}</SCPDURL><controlURL>{cms_ctl}</controlURL>",
            "<eventSubURL>{cms_evt}</eventSubURL></service>\n",
            "</serviceList></device></root>\n",
        ),
        dt = DEVICE_TYPE,
        name = escape(&dev.name),
        udn = escape(&dev.udn),
        cds = CDS_TYPE,
        cds_scpd = CDS_SCPD_PATH,
        cds_ctl = CDS_CONTROL_PATH,
        cds_evt = CDS_EVENT_PATH,
        cms = CMS_TYPE,
        cms_scpd = CMS_SCPD_PATH,
        cms_ctl = CMS_CONTROL_PATH,
        cms_evt = CMS_EVENT_PATH,
    )
}

/// `/cds.xml` and `/cms.xml`: an empty service description, as the Roku
/// accepted.
pub const SCPD: &str = "<?xml version=\"1.0\"?>\n<scpd xmlns=\"urn:schemas-upnp-org:service-1-0\"><specVersion><major>1</major><minor>0</minor></specVersion><actionList/><serviceStateTable/></scpd>\n";

fn didl(inner: &str) -> String {
    format!(
        "<DIDL-Lite xmlns=\"urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:upnp=\"urn:schemas-upnp-org:metadata-1-0/upnp/\">{inner}</DIDL-Lite>"
    )
}

/// DIDL-Lite of the live item.
#[must_use]
pub fn item(dev: &Device) -> String {
    format!(
        "<item id=\"{ITEM_ID}\" parentID=\"{ROOT_ID}\" restricted=\"1\"><dc:title>{}</dc:title><upnp:class>object.item.videoItem</upnp:class><res protocolInfo=\"{PROTOCOL_INFO}\" resolution=\"{RESOLUTION}\">{}</res></item>",
        escape(&dev.title),
        escape(&dev.stream_url()),
    )
}

/// DIDL-Lite of the root container.
#[must_use]
pub fn root_container(dev: &Device) -> String {
    format!(
        "<container id=\"{ROOT_ID}\" parentID=\"-1\" childCount=\"1\" restricted=\"1\"><dc:title>{}</dc:title><upnp:class>object.container</upnp:class></container>",
        escape(&dev.name),
    )
}

/// A control action this server answers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Action {
    Browse { object_id: String, flag: BrowseFlag },
    GetSortCapabilities,
    GetSearchCapabilities,
    GetSystemUpdateId,
    GetProtocolInfo,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BrowseFlag {
    Metadata,
    DirectChildren,
}

/// The service a control URL belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Service {
    ContentDirectory,
    ConnectionManager,
}

impl Service {
    #[must_use]
    pub fn urn(self) -> &'static str {
        match self {
            Self::ContentDirectory => CDS_TYPE,
            Self::ConnectionManager => CMS_TYPE,
        }
    }
}

/// A control request that gets a UPnP fault.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fault {
    /// 401 Invalid Action: unknown, or sent to the other service.
    InvalidAction,
    /// 402 Invalid Args: a Browse without a usable ObjectID or BrowseFlag.
    InvalidArgs,
    /// 701 No such object.
    NoSuchObject,
}

impl Fault {
    #[must_use]
    pub fn code(self) -> u16 {
        match self {
            Self::InvalidAction => 401,
            Self::InvalidArgs => 402,
            Self::NoSuchObject => 701,
        }
    }

    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::InvalidAction => "Invalid Action",
            Self::InvalidArgs => "Invalid Args",
            Self::NoSuchObject => "No such object",
        }
    }
}

/// The action name from a `SOAPACTION` header value:
/// `"urn:schemas-upnp-org:service:ContentDirectory:1#Browse"`, quoted or
/// not. The service type before `#` must be the one `service` implements.
#[must_use]
pub fn soap_action_name(header: &str, service: Service) -> Option<&str> {
    let value = header.trim();
    let value = value
        .strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(value);
    let (urn, name) = value.split_once('#')?;
    if urn != service.urn() || name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return None;
    }
    Some(name)
}

/// The text of the first `<tag>...</tag>` (tag written without a prefix
/// or attributes), XML-unescaped for the five named entities only.
fn element<'a>(body: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = body.find(&open)? + open.len();
    let len = body[start..].find(&close)?;
    Some(&body[start..start + len])
}

fn unescape(text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let tail = &rest[at..];
        let end = tail.find(';')?;
        out.push(match &tail[..=end] {
            "&amp;" => '&',
            "&lt;" => '<',
            "&gt;" => '>',
            "&quot;" => '"',
            "&apos;" => '\'',
            _ => return None,
        });
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    Some(out)
}

/// Parse a control request for `service`.
pub fn parse_action(service: Service, soap_action: &str, body: &str) -> Result<Action, Fault> {
    let name = soap_action_name(soap_action, service).ok_or(Fault::InvalidAction)?;
    match (service, name) {
        (Service::ContentDirectory, "Browse") => {
            let object_id = element(body, "ObjectID")
                .and_then(unescape)
                .ok_or(Fault::InvalidArgs)?;
            let flag = match element(body, "BrowseFlag").map(str::trim) {
                Some("BrowseMetadata") => BrowseFlag::Metadata,
                Some("BrowseDirectChildren") => BrowseFlag::DirectChildren,
                _ => return Err(Fault::InvalidArgs),
            };
            Ok(Action::Browse {
                object_id: object_id.trim().to_owned(),
                flag,
            })
        }
        (Service::ContentDirectory, "GetSortCapabilities") => Ok(Action::GetSortCapabilities),
        (Service::ContentDirectory, "GetSearchCapabilities") => Ok(Action::GetSearchCapabilities),
        (Service::ContentDirectory, "GetSystemUpdateID") => Ok(Action::GetSystemUpdateId),
        (Service::ConnectionManager, "GetProtocolInfo") => Ok(Action::GetProtocolInfo),
        _ => Err(Fault::InvalidAction),
    }
}

fn envelope(inner: &str) -> String {
    format!(
        "<?xml version=\"1.0\"?>\n<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\"><s:Body>{inner}</s:Body></s:Envelope>\n"
    )
}

fn response(service: Service, action: &str, body: &str) -> String {
    envelope(&format!(
        "<u:{action}Response xmlns:u=\"{}\">{body}</u:{action}Response>",
        service.urn()
    ))
}

/// The SOAP answer to an action, or the fault for it.
pub fn answer(dev: &Device, service: Service, action: &Action) -> Result<String, Fault> {
    Ok(match action {
        Action::Browse { object_id, flag } => {
            let (result, count) = match (object_id.as_str(), flag) {
                (ROOT_ID, BrowseFlag::Metadata) => (didl(&root_container(dev)), 1),
                (ROOT_ID, BrowseFlag::DirectChildren) => (didl(&item(dev)), 1),
                (ITEM_ID, BrowseFlag::Metadata) => (didl(&item(dev)), 1),
                (ITEM_ID, BrowseFlag::DirectChildren) => (didl(""), 0),
                _ => return Err(Fault::NoSuchObject),
            };
            response(
                service,
                "Browse",
                &format!(
                    "<Result>{}</Result><NumberReturned>{count}</NumberReturned><TotalMatches>{count}</TotalMatches><UpdateID>1</UpdateID>",
                    escape(&result)
                ),
            )
        }
        Action::GetSortCapabilities => {
            response(service, "GetSortCapabilities", "<SortCaps></SortCaps>")
        }
        Action::GetSearchCapabilities => response(
            service,
            "GetSearchCapabilities",
            "<SearchCaps></SearchCaps>",
        ),
        Action::GetSystemUpdateId => response(service, "GetSystemUpdateID", "<Id>1</Id>"),
        Action::GetProtocolInfo => response(
            service,
            "GetProtocolInfo",
            &format!("<Source>{PROTOCOL_INFO}</Source><Sink></Sink>"),
        ),
    })
}

/// A UPnP fault body (sent with `500 Internal Server Error`).
#[must_use]
pub fn fault(f: Fault) -> String {
    envelope(&format!(
        "<s:Fault><faultcode>s:Client</faultcode><faultstring>UPnPError</faultstring><detail><UPnPError xmlns=\"urn:schemas-upnp-org:control-1-0\"><errorCode>{}</errorCode><errorDescription>{}</errorDescription></UPnPError></detail></s:Fault>",
        f.code(),
        f.description()
    ))
}
