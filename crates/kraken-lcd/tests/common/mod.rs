//! Decode an 8-bit RGBA PNG for the render-once check.

use std::io::Cursor;

pub fn rgba_png_pixel(bytes: &[u8], x: u32, y: u32) -> Result<(u32, u32, [u8; 4]), String> {
    let decoded = decode_rgba(bytes)?;
    let index = (y as usize)
        .checked_mul(decoded.width as usize)
        .and_then(|row| row.checked_add(x as usize))
        .and_then(|pixel| pixel.checked_mul(4))
        .ok_or_else(|| "pixel index overflow".to_owned())?;
    let px = decoded
        .bytes
        .get(index..index + 4)
        .ok_or_else(|| "pixel out of range".to_owned())?;
    Ok((decoded.width, decoded.height, [px[0], px[1], px[2], px[3]]))
}

struct Decoded {
    width: u32,
    height: u32,
    bytes: Vec<u8>,
}

fn decode_rgba(bytes: &[u8]) -> Result<Decoded, String> {
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|err| err.to_string())?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| "png is too large".to_owned())?;
    let mut buffer = vec![0; size];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|err| err.to_string())?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "need 8-bit RGBA, got {:?} {:?}",
            info.bit_depth, info.color_type
        ));
    }
    buffer.truncate(info.buffer_size());
    Ok(Decoded {
        width: info.width,
        height: info.height,
        bytes: buffer,
    })
}
