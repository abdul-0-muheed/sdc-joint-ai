-- ============================================================
-- SDC-JOINT-AI  –  Complete Supabase Database Schema
-- Copy-paste this entire file into the Supabase SQL Editor
-- and click "Run" to create all tables.
-- ============================================================

-- ============================================================
-- 1. HELPDESK REQUESTS
-- ============================================================
CREATE TABLE IF NOT EXISTS helpdesk_requests (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    requester_name  TEXT        NOT NULL,
    requester_email TEXT        NOT NULL,
    phone_number    TEXT        NOT NULL,
    request_text    TEXT        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'Pending',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 2. STUDENT COMPLAINTS
-- ============================================================
CREATE TABLE IF NOT EXISTS student_complaints (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    student_name    TEXT        NOT NULL,
    student_email   TEXT        NOT NULL,
    department      TEXT,
    complaint_text  TEXT        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'Pending',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 3. CLUBS
-- ============================================================
CREATE TABLE IF NOT EXISTS clubs (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    club_name       TEXT        NOT NULL,
    description     TEXT,
    head_of_club    TEXT,
    contact_email   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 4. CLUB PROGRAMS
-- ============================================================
CREATE TABLE IF NOT EXISTS club_programs (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    club_id         BIGINT      NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
    program_name    TEXT        NOT NULL,
    description     TEXT,
    program_date    DATE        NOT NULL,
    start_time      TIME        NOT NULL,
    end_time        TIME        NOT NULL,
    venue           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 5. EVENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_name       TEXT        NOT NULL,
    description      TEXT,
    event_date       DATE        NOT NULL,
    start_time       TIME        NOT NULL,
    end_time         TIME        NOT NULL,
    venue            TEXT        NOT NULL,
    department       TEXT,
    registrationlink TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- 6. EXAM TIMETABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS exam_timetable (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    semester        TEXT        NOT NULL,
    subject_name    TEXT        NOT NULL,
    exam_date       DATE        NOT NULL,
    exam_time       TEXT        NOT NULL,
    venue           TEXT        NOT NULL,
    department      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- ============================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================
-- Enable RLS on all tables
ALTER TABLE helpdesk_requests  ENABLE ROW LEVEL SECURITY;
ALTER TABLE student_complaints ENABLE ROW LEVEL SECURITY;
ALTER TABLE clubs              ENABLE ROW LEVEL SECURITY;
ALTER TABLE club_programs      ENABLE ROW LEVEL SECURITY;
ALTER TABLE events             ENABLE ROW LEVEL SECURITY;
ALTER TABLE exam_timetable     ENABLE ROW LEVEL SECURITY;

-- -------------------------------------------------------
-- PUBLIC READ access for all tables (anon + authenticated)
-- -------------------------------------------------------
CREATE POLICY "Allow public read on helpdesk_requests"
    ON helpdesk_requests FOR SELECT
    USING (true);

CREATE POLICY "Allow public read on student_complaints"
    ON student_complaints FOR SELECT
    USING (true);

CREATE POLICY "Allow public read on clubs"
    ON clubs FOR SELECT
    USING (true);

CREATE POLICY "Allow public read on club_programs"
    ON club_programs FOR SELECT
    USING (true);

CREATE POLICY "Allow public read on events"
    ON events FOR SELECT
    USING (true);

CREATE POLICY "Allow public read on exam_timetable"
    ON exam_timetable FOR SELECT
    USING (true);

-- -------------------------------------------------------
-- PUBLIC INSERT for helpdesk & complaints (anyone can submit)
-- -------------------------------------------------------
CREATE POLICY "Allow public insert on helpdesk_requests"
    ON helpdesk_requests FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Allow public insert on student_complaints"
    ON student_complaints FOR INSERT
    WITH CHECK (true);

-- -------------------------------------------------------
-- AUTHENTICATED users can INSERT, UPDATE, DELETE on all tables
-- (admin management pages require login)
-- -------------------------------------------------------
CREATE POLICY "Allow authenticated insert on clubs"
    ON clubs FOR INSERT
    TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow authenticated update on clubs"
    ON clubs FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow authenticated delete on clubs"
    ON clubs FOR DELETE
    TO authenticated
    USING (true);

CREATE POLICY "Allow authenticated insert on club_programs"
    ON club_programs FOR INSERT
    TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow authenticated update on club_programs"
    ON club_programs FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow authenticated delete on club_programs"
    ON club_programs FOR DELETE
    TO authenticated
    USING (true);

CREATE POLICY "Allow authenticated insert on events"
    ON events FOR INSERT
    TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow authenticated update on events"
    ON events FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow authenticated delete on events"
    ON events FOR DELETE
    TO authenticated
    USING (true);

CREATE POLICY "Allow authenticated insert on exam_timetable"
    ON exam_timetable FOR INSERT
    TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow authenticated update on exam_timetable"
    ON exam_timetable FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow authenticated delete on exam_timetable"
    ON exam_timetable FOR DELETE
    TO authenticated
    USING (true);

-- Authenticated users can update complaints & helpdesk (status changes)
CREATE POLICY "Allow authenticated update on student_complaints"
    ON student_complaints FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow authenticated update on helpdesk_requests"
    ON helpdesk_requests FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);


-- ============================================================
-- EXAMPLE / SEED DATA
-- ============================================================

-- 1. Helpdesk Requests
INSERT INTO helpdesk_requests (requester_name, requester_email, phone_number, request_text, status) VALUES
('Rahul Sharma',   'rahul.sharma@example.com',   '+91 9876543210', 'I need help with my hostel room allocation. My room has not been assigned yet.', 'Pending'),
('Priya Nair',     'priya.nair@example.com',     '+91 9123456780', 'Please schedule a callback regarding fee payment options and scholarship details.', 'In Progress'),
('Amit Kumar',     'amit.kumar@example.com',     '+91 8765432109', 'I lost my student ID card. How can I get a replacement?', 'Resolved');

-- 2. Student Complaints
INSERT INTO student_complaints (student_name, student_email, department, complaint_text, status) VALUES
('Sneha Reddy',    'sneha.reddy@example.com',    'Computer Science & Engineering',             'The Wi-Fi in the CSE lab is extremely slow and disconnects frequently during online exams.', 'Pending'),
('Mohammed Faisal','faisal.m@example.com',        'Electronics & Communication Engineering',    'The air conditioning in the ECE seminar hall has not been working for the past two weeks.', 'In Progress'),
('Anjali Menon',   'anjali.menon@example.com',    'Master of Business Administration (MBA)',     'The library does not have the latest edition of the Management Accounting textbook.', 'Pending'),
('Kiran Rao',      'kiran.rao@example.com',       'Civil Engineering',                           'Street lights near the Civil Engineering block are not working, making it unsafe at night.', 'Resolved');

-- 3. Clubs
INSERT INTO clubs (club_name, description, head_of_club, contact_email) VALUES
('Coding Club',       'A community of passionate programmers who participate in hackathons, coding contests, and open-source projects.',  'Dr. Rajesh Bhat',     'codingclub@sdce.edu'),
('Robotics Club',     'Focuses on building robots, drones, and IoT projects. Participates in national-level robotics competitions.',       'Prof. Meena Shetty',  'robotics@sdce.edu'),
('Literary Club',     'Organizes debates, essay writing, poetry slams, and book reading sessions to foster literary skills.',              'Prof. Anand Kulkarni','literary@sdce.edu'),
('Sports Club',       'Manages all inter-college and intra-college sports events including cricket, football, basketball, and athletics.', 'Mr. Suresh Poojary',  'sports@sdce.edu'),
('Cultural Club',     'Organizes cultural festivals, dance, music, drama, and art exhibitions throughout the academic year.',              'Dr. Kavitha Rao',     'cultural@sdce.edu');

-- 4. Club Programs (references club IDs 1-5 from above)
INSERT INTO club_programs (club_id, program_name, description, program_date, start_time, end_time, venue) VALUES
(1, 'Code Sprint 2026',          'A 6-hour competitive coding marathon with prizes for top 3 teams.',                    '2026-03-15', '09:00', '15:00', 'CSE Lab 3'),
(1, 'Web Dev Workshop',          'Hands-on workshop on building full-stack web apps with Next.js and Supabase.',          '2026-03-22', '10:00', '13:00', 'Seminar Hall A'),
(2, 'Drone Racing Championship', 'Inter-college drone racing event with obstacle courses.',                               '2026-04-05', '08:00', '17:00', 'College Ground'),
(3, 'National Debate Competition','Annual debate competition on current affairs and technology ethics.',                   '2026-03-28', '09:30', '16:00', 'Auditorium'),
(4, 'Inter-College Cricket Tournament', 'T20 cricket tournament featuring 8 colleges from the region.',                   '2026-04-10', '08:00', '18:00', 'Sports Complex'),
(5, 'Rhythm & Blues Music Night', 'Live music performances by student bands and solo artists.',                            '2026-03-20', '18:00', '21:00', 'Open Air Theatre');

-- 5. Events
INSERT INTO events (event_name, description, event_date, start_time, end_time, venue, department, registrationlink) VALUES
('Tech Fest 2026',               'Annual technology festival with workshops, hackathons, project exhibitions, and guest lectures.', '2026-04-15', '09:00', '18:00', 'Main Campus',         'Computer Science & Engineering',             'forms.google.com/techfest2026'),
('Placement Drive - Infosys',    'Campus recruitment drive by Infosys for B.E. and MCA final year students.',                      '2026-03-10', '09:00', '17:00', 'Placement Cell',      'Training and Placement Department',          'forms.google.com/infosys-drive'),
('National Science Day Seminar', 'Guest lecture and panel discussion on AI in Healthcare by industry experts.',                    '2026-02-28', '10:00', '13:00', 'Auditorium',           NULL,                                          NULL),
('Annual Sports Day',            'Track and field events, team sports finals, and award ceremony.',                                '2026-03-25', '07:00', '17:00', 'Sports Complex',      NULL,                                          'forms.google.com/sportsday2026'),
('MBA Industry Visit',           'Industrial visit to Infosys Mangalore Development Center.',                                     '2026-03-18', '08:30', '16:00', 'Infosys Mangalore',   'Master of Business Administration (MBA)',      NULL);

-- 6. Exam Timetable
INSERT INTO exam_timetable (semester, subject_name, exam_date, exam_time, venue, department) VALUES
('Semester 6', 'Data Structures & Algorithms',       '2026-05-05', '9:00 AM - 12:00 PM',  'Exam Hall 1',  'Computer Science & Engineering'),
('Semester 6', 'Database Management Systems',         '2026-05-08', '9:00 AM - 12:00 PM',  'Exam Hall 1',  'Computer Science & Engineering'),
('Semester 6', 'Computer Networks',                   '2026-05-12', '9:00 AM - 12:00 PM',  'Exam Hall 2',  'Computer Science & Engineering'),
('Semester 4', 'Digital Electronics',                  '2026-05-06', '2:00 PM - 5:00 PM',   'Exam Hall 3',  'Electronics & Communication Engineering'),
('Semester 4', 'Signals & Systems',                   '2026-05-10', '2:00 PM - 5:00 PM',   'Exam Hall 3',  'Electronics & Communication Engineering'),
('Semester 2', 'Engineering Mathematics II',           '2026-05-05', '2:00 PM - 5:00 PM',   'Exam Hall 4',  'Civil Engineering'),
('Semester 2', 'Engineering Physics',                  '2026-05-09', '9:00 AM - 12:00 PM',  'Exam Hall 4',  'Aeronautical Engineering'),
('Semester 4', 'Strategic Management',                 '2026-05-07', '10:00 AM - 1:00 PM',  'MBA Block',    'Master of Business Administration (MBA)');
