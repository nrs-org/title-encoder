-- MB artist -> YouTube channel-id links, for the artist-wrapper data (v7).
-- Leakage filtering happens client-side: only artists present in the
-- already-filtered raw/artist_names.csv are kept (channels.py).
\set ON_ERROR_STOP 1
set search_path = musicbrainz;
\copy (select l.entity0 as artist, u.url from l_artist_url l join url u on u.id = l.entity1 where u.url ~ '^https://(www\.|music\.)?youtube\.com/channel/UC[A-Za-z0-9_-]{22}' order by 1) to '/tmp/te_channels.csv' with (format csv, header)
