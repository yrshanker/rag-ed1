from __future__ import annotations

import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from textwrap import dedent


def generate_imscc(
    out_path: Path | str,
    *,
    title: str = "Minimal CC Example",
    description: str = "A tiny Common Cartridge for testing.",
    version: str = "1.3.0",
    mode: str = "webcontent",
    weblink_url: str = "https://example.org",
    language: str = "en-US",
) -> Path:
    """Generate a small IMS Common Cartridge file.

    Parameters
    ----------
    out_path : Path | str
        Destination ``.imscc`` path.
    title : str, optional
        Human-readable title.
    description : str, optional
        Manifest-level description.
    version : str, optional
        Cartridge version string, e.g. ``"1.3.0"``.
    mode : str, optional
        ``"webcontent"`` for a thick cartridge or ``"thin_weblink"`` for a Thin CC
        web link.
    weblink_url : str, optional
        Target URL when ``mode="thin_weblink"``.
    language : str, optional
        Metadata language tag.

    Returns
    -------
    Path
        Path to the generated ``.imscc`` file.
    """
    assert mode in {"webcontent", "thin_weblink"}

    path = Path(out_path).with_suffix(".imscc")
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest_id = f"M_{uuid.uuid4().hex[:8]}"
    org_id = f"O_{uuid.uuid4().hex[:8]}"
    item_id = f"I_{uuid.uuid4().hex[:8]}"

    if mode == "webcontent":
        res_id = f"RES_{uuid.uuid4().hex[:8]}"
        web_folder = "webcontent"
        html_rel = f"{web_folder}/index.html"
        resources_xml = dedent(
            f"""
            <resources>
              <resource identifier="{res_id}" type="webcontent" href="{html_rel}">
                <file href="{html_rel}"/>
              </resource>
            </resources>
            """
        ).strip()
    else:
        res_id = f"WL_{uuid.uuid4().hex[:8]}"
        wl_folder = "weblinks"
        wl_file = f"{wl_folder}/weblink1.xml"
        resources_xml = dedent(
            f"""
            <resources>
              <resource identifier="{res_id}" type="imswl_xmlv1p0" href="{wl_file}">
                <file href="{wl_file}"/>
              </resource>
            </resources>
            """
        ).strip()

    organizations_xml = dedent(
        f"""
        <organizations>
          <organization identifier="{org_id}" structure="rooted-hierarchy">
            <title>{_escape_xml(title)}</title>
            <item identifier="{item_id}" identifierref="{res_id}">
              <title>{_escape_xml(title)}</title>
            </item>
          </organization>
        </organizations>
        """
    ).strip()

    lom_ns = (
        "http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest"
        if version.startswith("1.2")
        else "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"
    )
    created = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata_xml = dedent(
        f"""
        <metadata>
          <schema>1EdTech Common Cartridge</schema>
          <schemaversion>{version}</schemaversion>
          <lomimscc:lom xmlns:lomimscc="{lom_ns}">
            <lomimscc:general>
              <lomimscc:title>
                <lomimscc:string language="{language}">{_escape_xml(title)}</lomimscc:string>
              </lomimscc:title>
              <lomimscc:description>
                <lomimscc:string language="{language}">{_escape_xml(description)}</lomimscc:string>
              </lomimscc:description>
              <lomimscc:keyword>
                <lomimscc:string language="{language}">sample</lomimscc:string>
              </lomimscc:keyword>
              <lomimscc:coverage>
                <lomimscc:string language="{language}">{created}</lomimscc:string>
              </lomimscc:coverage>
            </lomimscc:general>
          </lomimscc:lom>
        </metadata>
        """
    ).strip()

    manifest_xml = (
        dedent(
            f"""
            <?xml version="1.0" encoding="UTF-8"?>
            <manifest
              identifier="{manifest_id}"
              xmlns="http://www.imsglobal.org/xsd/imscc/imscp_v1p1"
              xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
              {metadata_xml}
              {organizations_xml}
              {resources_xml}
            </manifest>
            """
        ).strip()
        + "\n"
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest_xml)
        if mode == "webcontent":
            html = dedent(
                f"""
                <!doctype html>
                <html lang="{language}">
                  <head><meta charset="utf-8"><title>{_escape_html(title)}</title></head>
                  <body>
                    <h1>{_escape_html(title)}</h1>
                    <p>This is a minimal Common Cartridge web page for testing import.</p>
                  </body>
                </html>
                """
            )
            zf.writestr(html_rel, html)
        else:
            wl_xml = dedent(
                f"""
                <?xml version="1.0" encoding="UTF-8"?>
                <wl:webLink xmlns:wl="http://www.imsglobal.org/xsd/imswl_v1p0"
                            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                  <title>{_escape_xml(title)}</title>
                  <url href="{_escape_xml(weblink_url)}" target="_parent" windowFeatures=""/>
                </wl:webLink>
                """
            )
            zf.writestr(wl_file, wl_xml)

    return path


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _escape_html(text: str) -> str:
    return _escape_xml(text).replace("'", "&#39;")
