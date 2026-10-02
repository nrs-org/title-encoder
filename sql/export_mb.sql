-- Export JP/EN name groups from the local MusicBrainz mirror for title-encoder
-- training. Run by build_mb.py inside the db container; everything is TEMP, so
-- the mirror itself is never modified.
--
-- Inputs (copied in by build_mb.py):
--   /tmp/te/lib_mbid.tsv          kind<TAB>gid   — MB entities in the eval library
--   /tmp/te/lib_artist_names.tsv  lower(name)    — every alias of a library artist
-- Outputs: /tmp/te/out/*.csv, one row per (group, name).

\set ON_ERROR_STOP 1
set search_path = musicbrainz;
set work_mem = '512MB';

-- ── Leakage inputs ───────────────────────────────────────────────────────────
create temp table lib_mbid (kind text, gid uuid);
\copy lib_mbid from '/tmp/te/lib_mbid.tsv'
create temp table lib_artist_name (name text primary key);
\copy lib_artist_name from '/tmp/te/lib_artist_names.tsv'

-- Placeholder credits that would glue unrelated songs into one group.
create temp table special_artist as
select id from artist where gid in (
    '89ad4ac3-39f7-470e-963a-56509c546377',  -- Various Artists
    '125ec42a-7229-4250-afc5-e057484327fe'   -- [unknown]
);

-- ── Tiers ────────────────────────────────────────────────────────────────────
create temp table jp_area as
with jp as (select a.id from iso_3166_1 i join area a on a.id = i.area where i.code = 'JP')
select id from jp union select descendant from area_containment where parent in (select id from jp);

create temp table en_area as
with en as (select a.id from iso_3166_1 i join area a on a.id = i.area
            where i.code in ('US', 'GB', 'CA', 'AU', 'IE', 'NZ'))
select id from en union select descendant from area_containment where parent in (select id from en);

create temp table lang_artist as
select acn.artist, count(*) filter (where l.iso_code_3 = 'jpn') as jpn,
       count(*) filter (where l.iso_code_3 = 'eng') as eng
from release r
join language l on l.id = r.language
join artist_credit_name acn on acn.artist_credit = r.artist_credit
where l.iso_code_3 in ('jpn', 'eng')
group by acn.artist;

-- Area decides when set. Release language only decides for artists with no
-- area at all (doujin / Vocaloid producers often lack one): Japanese reissues
-- of Western albums carry language 'jpn', so it can't override a known area.
create temp table artist_tier (id integer primary key, tier text not null);
insert into artist_tier
select id, tier from (
    select a.id,
           case when a.area in (select id from jp_area) or a.begin_area in (select id from jp_area) then 'jp'
                when a.area is not null or a.begin_area is not null then
                     case when a.area in (select id from en_area) or a.begin_area in (select id from en_area)
                               or coalesce(la.eng, 0) > 0 then 'en' end
                when coalesce(la.jpn, 0) > 0 and la.jpn >= coalesce(la.eng, 0) then 'jp'
                when coalesce(la.eng, 0) > 0 then 'en'
           end as tier
    from artist a
    left join lang_artist la on la.artist = a.id
    where a.id not in (select id from special_artist)
) t where tier is not null;
analyze artist_tier;

-- ── Leakage exclusion: library entities, their credited artists, and every
-- artist whose name matches a library artist alias ────────────────────────────
create temp table excl_artist (id integer primary key);
insert into excl_artist
select id from artist where gid in (select gid from lib_mbid where kind = 'artist')
union select acn.artist from recording x join artist_credit_name acn on acn.artist_credit = x.artist_credit
      where x.gid in (select gid from lib_mbid where kind = 'recording')
union select acn.artist from release x join artist_credit_name acn on acn.artist_credit = x.artist_credit
      where x.gid in (select gid from lib_mbid where kind = 'release')
union select acn.artist from release_group x join artist_credit_name acn on acn.artist_credit = x.artist_credit
      where x.gid in (select gid from lib_mbid where kind = 'release-group')
union select id from artist where lower(name) in (select name from lib_artist_name)
union select artist from artist_alias where lower(name) in (select name from lib_artist_name)
union select artist from artist_credit_name where lower(name) in (select name from lib_artist_name);
delete from excl_artist where id in (select id from special_artist);
analyze excl_artist;

-- Per artist credit: tier (jp wins over en) and whether any credited artist is
-- excluded. Credits with no tiered artist are dropped.
create temp table ac_tier (artist_credit integer primary key, tier text not null, excl boolean not null);
insert into ac_tier
select acn.artist_credit,
       case when bool_or(t.tier = 'jp') then 'jp' else 'en' end,
       bool_or(acn.artist in (select id from excl_artist))
from artist_credit_name acn
join artist_tier t on t.id = acn.artist
group by acn.artist_credit;
analyze ac_tier;

create temp table excl_recording (id integer primary key);
insert into excl_recording
select r.id from recording r join ac_tier t on t.artist_credit = r.artist_credit where t.excl
union select id from recording where gid in (select gid from lib_mbid where kind = 'recording');
analyze excl_recording;

create temp table excl_work (id integer primary key);
insert into excl_work
select distinct entity1 from l_recording_work where entity0 in (select id from excl_recording);
analyze excl_work;

-- ── Song groups: recordings of one work by one artist credit (audio + video
-- recordings of the same song, plus its version variants) ─────────────────────
create temp table song_rec as
select lrw.entity1 as work, r.artist_credit, t.tier, r.id as recording, r.name, r.video, r.length
from recording r
join l_recording_work lrw on lrw.entity0 = r.id
join ac_tier t on t.artist_credit = r.artist_credit
where not t.excl
  and r.artist_credit not in (select artist_credit from artist_credit_name where artist in (select id from special_artist))
  and lrw.entity1 not in (select id from excl_work)
  and (t.tier = 'jp' or lrw.entity1 % 4 = 0);

\copy (select s.*, ac.name as credit from song_rec s join artist_credit ac on ac.id = s.artist_credit join (select work, artist_credit from song_rec group by 1, 2 having count(distinct lower(name)) > 1 or (bool_or(video) and bool_or(not video))) g using (work, artist_credit) order by work, artist_credit, recording) to '/tmp/te/out/song_recordings.csv' with (format csv, header)

-- Every JP / sampled-EN song (one row per work x credit, even single-name
-- ones): the clean titles the synthetic video-title set is generated from.
\copy (select work, artist_credit, tier, min(name) as name, (select name from artist_credit where id = artist_credit) as credit from song_rec group by work, artist_credit, tier order by work, artist_credit) to '/tmp/te/out/songs.csv' with (format csv, header)

-- Credit -> artists (position 0 = primary), for artist-conditioned data:
-- cover negatives need disjoint artists, and aliases vary the artist slot.
\copy (select acn.artist_credit, acn.position, acn.artist from artist_credit_name acn where acn.artist_credit in (select distinct artist_credit from song_rec) order by 1, 2) to '/tmp/te/out/credit_artists.csv' with (format csv, header)

-- Work tier: jp if any non-excluded jp recording performs it.
create temp table work_tier as
select work, case when bool_or(tier = 'jp') then 'jp' else 'en' end as tier
from song_rec group by work;

-- ── Alias groups (name + every alias; alias ↔ alias pairs come from these) ────
\copy (select g.* from (select r.id as entity, t.tier, r.name, 'name' as kind, null::text as locale from recording r join ac_tier t on t.artist_credit = r.artist_credit where not t.excl and r.id in (select recording from recording_alias) and (t.tier = 'jp' or r.id % 2 = 0) union all select ra.recording, t.tier, ra.name, 'alias', ra.locale from recording_alias ra join recording r on r.id = ra.recording join ac_tier t on t.artist_credit = r.artist_credit where not t.excl and (t.tier = 'jp' or r.id % 2 = 0)) g order by entity) to '/tmp/te/out/recording_names.csv' with (format csv, header)

\copy (select g.* from (select w.id as entity, wt.tier, w.name, 'name' as kind, null::text as locale from work w join work_tier wt on wt.work = w.id where w.id in (select work from work_alias) union all select wa.work, wt.tier, wa.name, 'alias', wa.locale from work_alias wa join work_tier wt on wt.work = wa.work) g order by entity) to '/tmp/te/out/work_names.csv' with (format csv, header)

\copy (select g.* from (select a.id as entity, t.tier, a.name, 'name' as kind, null::text as locale from artist a join artist_tier t on t.id = a.id where a.id not in (select id from excl_artist) and (t.tier = 'jp' or a.id % 4 = 0) union all select aa.artist, t.tier, aa.name, 'alias', aa.locale from artist_alias aa join artist_tier t on t.id = aa.artist where aa.artist not in (select id from excl_artist) and (t.tier = 'jp' or aa.artist % 4 = 0) union all select distinct acn.artist, t.tier, acn.name, 'credited', null from artist_credit_name acn join artist_tier t on t.id = acn.artist where acn.artist not in (select id from excl_artist) and (t.tier = 'jp' or acn.artist % 4 = 0)) g order by entity) to '/tmp/te/out/artist_names.csv' with (format csv, header)

\copy (select g.* from (select rg.id as entity, t.tier, rg.name, 'name' as kind, null::text as locale from release_group rg join ac_tier t on t.artist_credit = rg.artist_credit where not t.excl and rg.gid not in (select gid from lib_mbid where kind = 'release-group') and (t.tier = 'jp' or rg.id % 4 = 0) union all select rga.release_group, t.tier, rga.name, 'alias', rga.locale from release_group_alias rga join release_group rg on rg.id = rga.release_group join ac_tier t on t.artist_credit = rg.artist_credit where not t.excl and (t.tier = 'jp' or rg.id % 4 = 0) union all select distinct r.release_group, t.tier, r.name, 'release', null from release r join release_group rg on rg.id = r.release_group join ac_tier t on t.artist_credit = rg.artist_credit where not t.excl and (t.tier = 'jp' or rg.id % 4 = 0)) g order by entity) to '/tmp/te/out/release_group_names.csv' with (format csv, header)

-- ── MB-linked YouTube videos (titles fetched separately by fetch_youtube.py;
-- the source of real video-title templates and noisy-title positives) ────────
\copy (select r.id as recording, r.name, ac.name as credit, t.tier, u.url from l_recording_url l join url u on u.id = l.entity1 join recording r on r.id = l.entity0 join ac_tier t on t.artist_credit = r.artist_credit join artist_credit ac on ac.id = r.artist_credit where not t.excl and (u.url like 'https://www.youtube.com/watch?v=%' or u.url like 'https://youtu.be/%') and (t.tier = 'jp' or r.id % 8 = 0) order by r.id) to '/tmp/te/out/video_links.csv' with (format csv, header)

-- ── Counts for the dataset card ──────────────────────────────────────────────
\copy (select 'artist_tier_jp', count(*) from artist_tier where tier = 'jp' union all select 'artist_tier_en', count(*) from artist_tier where tier = 'en' union all select 'excl_artist', count(*) from excl_artist union all select 'excl_recording', count(*) from excl_recording union all select 'excl_work', count(*) from excl_work union all select 'lib_mbid', count(*) from lib_mbid union all select 'lib_artist_name', count(*) from lib_artist_name) to '/tmp/te/out/counts.csv' with (format csv)
