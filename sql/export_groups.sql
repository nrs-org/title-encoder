-- Group / agency names for the `{g}` slot of synthetic artist wrappers (v7):
-- labels and group-type artists located in Japan. Names only; library
-- affiliations are removed client-side (artist_wrap.py).
\set ON_ERROR_STOP 1
set search_path = musicbrainz;
create temp table jp_area as
with jp as (select a.id from iso_3166_1 i join area a on a.id = i.area where i.code = 'JP')
select id from jp union select descendant from area_containment where parent in (select id from jp);
\copy (select 'label' as kind, name from label where area in (select id from jp_area) union select 'group', name from artist where type = 2 and (area in (select id from jp_area) or begin_area in (select id from jp_area))) to '/tmp/te_groups.csv' with (format csv, header)
