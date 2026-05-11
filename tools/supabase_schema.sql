-- FocusGuardian Supabase schema.
-- Run this once in Supabase SQL Editor before filling supabase_url/api_key in config.json.
-- The permissive policies below are for the current desktop-app storage model.

create table if not exists public.focusguardian_users (
  user_id text primary key,
  username text not null unique,
  password_hash text not null,
  created_at bigint not null default 0,
  created_at_iso text not null default '',
  last_login_at bigint not null default 0,
  last_login_at_iso text not null default '',
  is_active boolean not null default true,
  profile_name text not null default 'default',
  raw_payload jsonb not null default '{}'::jsonb
);

create table if not exists public.focusguardian_sessions (
  id bigserial primary key,
  timestamp bigint,
  timestamp_iso text,
  profile_name text,
  session_seconds double precision,
  focus_seconds double precision,
  focus_seconds_raw double precision,
  focus_seconds_cleaned double precision,
  distraction_count integer,
  break_count integer,
  avg_score double precision,
  avg_score_raw double precision,
  avg_score_cleaned double precision,
  min_score double precision,
  max_score double precision,
  blink_rate_per_min double precision,
  avg_ear double precision,
  eye_closure_ratio double precision,
  perclos double precision,
  fatigue_onset_minutes double precision,
  score_drop_per_hour double precision,
  score_drop_per_hour_raw double precision,
  score_drop_per_hour_cleaned double precision,
  uncertain_seconds_raw double precision,
  uncertain_seconds_cleaned double precision,
  uncertain_measurement_noise_seconds double precision,
  uncertain_behavioral_seconds double precision,
  analytics_quality_score double precision,
  session_quality_weight double precision,
  face_presence_ratio double precision,
  minutes_since_last_break double precision,
  work_interval_minutes_used integer,
  break_duration_minutes_used integer,
  state_on_screen double precision,
  state_writing double precision,
  state_phone double precision,
  state_drowsy double precision,
  state_away double precision,
  state_uncertain double precision,
  session_exit_reason text,
  session_exit_reason_label text,
  session_exit_focus_rating double precision,
  session_exit_focus_rating_label text,
  session_exit_note text,
  raw_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.focusguardian_user_baselines (
  profile_name text primary key,
  updated_at bigint,
  updated_at_iso text,
  session_count integer,
  personalization_weight double precision,
  adaptation_stage text,
  blink_rate_baseline double precision,
  avg_ear_baseline double precision,
  eye_closure_ratio_baseline double precision,
  perclos_baseline double precision,
  average_focus_score_baseline double precision,
  average_distraction_density double precision,
  average_fatigue_onset_minutes double precision,
  focus_score_decay_per_hour double precision,
  recommended_work_minutes integer,
  recommended_break_minutes integer,
  last_quality_score double precision,
  raw_payload jsonb not null default '{}'::jsonb
);

create table if not exists public.focusguardian_focus_events (
  id bigserial primary key,
  timestamp bigint,
  timestamp_iso text,
  profile_name text,
  session_id text,
  event_type text,
  event_count integer,
  event_seconds double precision,
  avg_confidence double precision,
  metadata jsonb,
  raw_payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.focusguardian_profile_settings (
  profile_name text primary key,
  updated_at bigint,
  updated_at_iso text,
  settings_json text,
  settings jsonb not null default '{}'::jsonb
);

create index if not exists focusguardian_sessions_profile_timestamp_idx
  on public.focusguardian_sessions (profile_name, timestamp desc);

create index if not exists focusguardian_focus_events_profile_timestamp_idx
  on public.focusguardian_focus_events (profile_name, timestamp desc);

grant usage on schema public to anon, authenticated;
grant select, insert, update on
  public.focusguardian_users,
  public.focusguardian_sessions,
  public.focusguardian_user_baselines,
  public.focusguardian_focus_events,
  public.focusguardian_profile_settings
to anon, authenticated;
grant usage, select on all sequences in schema public to anon, authenticated;

alter table public.focusguardian_users enable row level security;
alter table public.focusguardian_sessions enable row level security;
alter table public.focusguardian_user_baselines enable row level security;
alter table public.focusguardian_focus_events enable row level security;
alter table public.focusguardian_profile_settings enable row level security;

drop policy if exists focusguardian_users_select on public.focusguardian_users;
drop policy if exists focusguardian_users_insert on public.focusguardian_users;
drop policy if exists focusguardian_users_update on public.focusguardian_users;
create policy focusguardian_users_select on public.focusguardian_users for select to anon, authenticated using (true);
create policy focusguardian_users_insert on public.focusguardian_users for insert to anon, authenticated with check (true);
create policy focusguardian_users_update on public.focusguardian_users for update to anon, authenticated using (true) with check (true);

drop policy if exists focusguardian_sessions_select on public.focusguardian_sessions;
drop policy if exists focusguardian_sessions_insert on public.focusguardian_sessions;
create policy focusguardian_sessions_select on public.focusguardian_sessions for select to anon, authenticated using (true);
create policy focusguardian_sessions_insert on public.focusguardian_sessions for insert to anon, authenticated with check (true);

drop policy if exists focusguardian_user_baselines_select on public.focusguardian_user_baselines;
drop policy if exists focusguardian_user_baselines_insert on public.focusguardian_user_baselines;
drop policy if exists focusguardian_user_baselines_update on public.focusguardian_user_baselines;
create policy focusguardian_user_baselines_select on public.focusguardian_user_baselines for select to anon, authenticated using (true);
create policy focusguardian_user_baselines_insert on public.focusguardian_user_baselines for insert to anon, authenticated with check (true);
create policy focusguardian_user_baselines_update on public.focusguardian_user_baselines for update to anon, authenticated using (true) with check (true);

drop policy if exists focusguardian_focus_events_select on public.focusguardian_focus_events;
drop policy if exists focusguardian_focus_events_insert on public.focusguardian_focus_events;
create policy focusguardian_focus_events_select on public.focusguardian_focus_events for select to anon, authenticated using (true);
create policy focusguardian_focus_events_insert on public.focusguardian_focus_events for insert to anon, authenticated with check (true);

drop policy if exists focusguardian_profile_settings_select on public.focusguardian_profile_settings;
drop policy if exists focusguardian_profile_settings_insert on public.focusguardian_profile_settings;
drop policy if exists focusguardian_profile_settings_update on public.focusguardian_profile_settings;
create policy focusguardian_profile_settings_select on public.focusguardian_profile_settings for select to anon, authenticated using (true);
create policy focusguardian_profile_settings_insert on public.focusguardian_profile_settings for insert to anon, authenticated with check (true);
create policy focusguardian_profile_settings_update on public.focusguardian_profile_settings for update to anon, authenticated using (true) with check (true);
