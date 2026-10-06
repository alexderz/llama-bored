//! The MediaServer documents and ContentDirectory answers (goldens).

mod common;

use common::{device, fixture};
use llama_cast::dlna::{
    self, Action, BrowseFlag, DLNA_FEATURES, Fault, PROTOCOL_INFO, STREAM_MIME, Service,
};

const CDS: &str = "\"urn:schemas-upnp-org:service:ContentDirectory:1#Browse\"";

/// A Browse body as the Roku sends it (Filter list included, and ignored).
fn browse(object_id: &str, flag: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?><s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\"><s:Body><u:Browse xmlns:u=\"urn:schemas-upnp-org:service:ContentDirectory:1\"><ObjectID>{object_id}</ObjectID><BrowseFlag>{flag}</BrowseFlag><Filter>dc:title,upnp:class,res,res@resolution,res@duration,upnp:albumArtURI</Filter><StartingIndex>0</StartingIndex><RequestedCount>100</RequestedCount><SortCriteria></SortCriteria></u:Browse></s:Body></s:Envelope>"
    )
}

fn browse_answer(object_id: &str, flag: &str) -> Result<String, Fault> {
    let action = dlna::parse_action(Service::ContentDirectory, CDS, &browse(object_id, flag))?;
    dlna::answer(&device(), Service::ContentDirectory, &action)
}

#[test]
fn protocol_info_is_the_roku_variant_exactly() {
    assert_eq!(
        PROTOCOL_INFO,
        "http-get:*:video/mpeg:DLNA.ORG_PN=MPEG_TS_HD_NA_ISO;DLNA.ORG_OP=00;DLNA.ORG_FLAGS=01700000000000000000000000000000"
    );
    assert_eq!(
        PROTOCOL_INFO,
        format!("http-get:*:{STREAM_MIME}:{DLNA_FEATURES}")
    );
    // video/mp2t made the Roku say "No compatible videos found".
    assert!(!PROTOCOL_INFO.contains("mp2t"));
    assert_eq!(dlna::RESOLUTION, "1920x1080");
    assert_eq!(dlna::STREAM_PATH, "/live.ts");
}

/// #20: the live item is streaming-only: no seek operations, the
/// streaming transfer mode bit, and no byte/time range or sender pacing.
#[test]
fn dlna_flags_describe_a_live_stream() {
    let field = |key: &str| {
        DLNA_FEATURES
            .split(';')
            .find_map(|kv| kv.strip_prefix(key))
            .unwrap_or_else(|| panic!("{key} missing"))
    };
    assert_eq!(field("DLNA.ORG_PN="), "MPEG_TS_HD_NA_ISO");
    assert_eq!(field("DLNA.ORG_OP="), "00", "no time or byte seek");
    let flags = field("DLNA.ORG_FLAGS=");
    assert_eq!(flags.len(), 32);
    assert!(flags[8..].bytes().all(|b| b == b'0'), "reserved bits set");
    let primary = u32::from_str_radix(&flags[..8], 16).unwrap();
    assert_eq!(
        primary,
        dlna::FLAG_STREAMING
            | dlna::FLAG_BACKGROUND
            | dlna::FLAG_HTTP_STALLING
            | dlna::FLAG_DLNA_V15
    );
    for (bit, what) in [
        (31, "sender paced"),
        (30, "lop-npt (time seek)"),
        (29, "lop-bytes (byte seek)"),
        (23, "interactive transfer"),
    ] {
        assert_eq!(primary & (1 << bit), 0, "{what} must be clear");
    }
}

#[test]
fn description_golden() {
    let want = std::fs::read_to_string(fixture("desc.xml")).unwrap();
    assert_eq!(dlna::description(&device()), want);
}

#[test]
fn friendly_name_is_escaped() {
    let mut dev = device();
    dev.name = "Den <tty11> & \"co\"".to_owned();
    let desc = dlna::description(&dev);
    assert!(desc.contains("<friendlyName>Den &lt;tty11&gt; &amp; &quot;co&quot;</friendlyName>"));
    let item = dlna::item(&dev);
    assert!(item.contains("<dc:title>Den &lt;tty11&gt; &amp; &quot;co&quot; live</dc:title>"));
}

#[test]
fn browse_direct_children_of_root_golden() {
    let want = std::fs::read_to_string(fixture("browse-children.xml")).unwrap();
    assert_eq!(browse_answer("0", "BrowseDirectChildren").unwrap(), want);
}

#[test]
fn browse_metadata_of_root_and_item() {
    let root = browse_answer("0", "BrowseMetadata").unwrap();
    assert!(root.contains("&lt;container id=&quot;0&quot; parentID=&quot;-1&quot; childCount=&quot;1&quot; restricted=&quot;1&quot;&gt;&lt;dc:title&gt;llama-bored&lt;/dc:title&gt;&lt;upnp:class&gt;object.container&lt;/upnp:class&gt;&lt;/container&gt;"), "{root}");
    assert!(root.contains("<NumberReturned>1</NumberReturned><TotalMatches>1</TotalMatches>"));
    let item = browse_answer("1", "BrowseMetadata").unwrap();
    let children = browse_answer("0", "BrowseDirectChildren").unwrap();
    assert_eq!(
        item, children,
        "the item's metadata is the root's one child"
    );
    let none = browse_answer("1", "BrowseDirectChildren").unwrap();
    assert!(none.contains("<NumberReturned>0</NumberReturned><TotalMatches>0</TotalMatches>"));
}

#[test]
fn filter_and_paging_are_ignored() {
    let plain = "<u:Browse><ObjectID>0</ObjectID><BrowseFlag>BrowseDirectChildren</BrowseFlag><Filter>*</Filter><StartingIndex>5</StartingIndex></u:Browse>";
    let action = dlna::parse_action(Service::ContentDirectory, CDS, plain).unwrap();
    assert_eq!(
        action,
        Action::Browse {
            object_id: "0".into(),
            flag: BrowseFlag::DirectChildren
        }
    );
    assert_eq!(
        dlna::answer(&device(), Service::ContentDirectory, &action).unwrap(),
        browse_answer("0", "BrowseDirectChildren").unwrap()
    );
}

#[test]
fn browse_faults() {
    assert_eq!(
        browse_answer("2", "BrowseMetadata"),
        Err(Fault::NoSuchObject)
    );
    assert_eq!(
        browse_answer("0", "BrowseEverything"),
        Err(Fault::InvalidArgs)
    );
    assert_eq!(
        dlna::parse_action(
            Service::ContentDirectory,
            CDS,
            "<BrowseFlag>BrowseMetadata</BrowseFlag>"
        ),
        Err(Fault::InvalidArgs)
    );
    // An unknown entity is not guessed at.
    assert_eq!(
        dlna::parse_action(
            Service::ContentDirectory,
            CDS,
            "<ObjectID>&bogus;</ObjectID><BrowseFlag>BrowseMetadata</BrowseFlag>"
        ),
        Err(Fault::InvalidArgs)
    );
    let fault = dlna::fault(Fault::NoSuchObject);
    assert!(
        fault.contains(
            "<errorCode>701</errorCode><errorDescription>No such object</errorDescription>"
        )
    );
}

#[test]
fn soap_action_header_forms() {
    for header in [
        "\"urn:schemas-upnp-org:service:ContentDirectory:1#Browse\"",
        "urn:schemas-upnp-org:service:ContentDirectory:1#Browse",
        "  \"urn:schemas-upnp-org:service:ContentDirectory:1#Browse\" ",
    ] {
        assert_eq!(
            dlna::soap_action_name(header, Service::ContentDirectory),
            Some("Browse"),
            "{header}"
        );
    }
    for header in [
        "",
        "Browse",
        "\"urn:schemas-upnp-org:service:ConnectionManager:1#Browse\"",
        "\"urn:schemas-upnp-org:service:ContentDirectory:1#\"",
        "\"urn:schemas-upnp-org:service:ContentDirectory:1#Br owse\"",
    ] {
        assert_eq!(
            dlna::soap_action_name(header, Service::ContentDirectory),
            None,
            "{header}"
        );
    }
}

#[test]
fn other_actions() {
    let dev = device();
    let ask = |service: Service, name: &str| {
        let header = format!("\"{}#{name}\"", service.urn());
        dlna::parse_action(service, &header, "").and_then(|a| dlna::answer(&dev, service, &a))
    };
    let sort = ask(Service::ContentDirectory, "GetSortCapabilities").unwrap();
    assert!(sort.contains("<u:GetSortCapabilitiesResponse xmlns:u=\"urn:schemas-upnp-org:service:ContentDirectory:1\"><SortCaps></SortCaps></u:GetSortCapabilitiesResponse>"));
    let search = ask(Service::ContentDirectory, "GetSearchCapabilities").unwrap();
    assert!(search.contains("<SearchCaps></SearchCaps>"));
    let update = ask(Service::ContentDirectory, "GetSystemUpdateID").unwrap();
    assert!(update.contains("<u:GetSystemUpdateIDResponse xmlns:u=\"urn:schemas-upnp-org:service:ContentDirectory:1\"><Id>1</Id></u:GetSystemUpdateIDResponse>"));
    let proto = ask(Service::ConnectionManager, "GetProtocolInfo").unwrap();
    assert!(proto.contains(&format!(
        "<u:GetProtocolInfoResponse xmlns:u=\"urn:schemas-upnp-org:service:ConnectionManager:1\"><Source>{PROTOCOL_INFO}</Source><Sink></Sink></u:GetProtocolInfoResponse>"
    )));
    // Each action belongs to its own service.
    assert_eq!(
        ask(Service::ConnectionManager, "Browse"),
        Err(Fault::InvalidAction)
    );
    assert_eq!(
        ask(Service::ContentDirectory, "GetProtocolInfo"),
        Err(Fault::InvalidAction)
    );
    assert_eq!(
        ask(Service::ContentDirectory, "DestroyObject"),
        Err(Fault::InvalidAction)
    );
}
