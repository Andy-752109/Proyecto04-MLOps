ALTER TABLE `training_jobs` ADD `run_kind` enum('smoke','campaign') NOT NULL;--> statement-breakpoint
ALTER TABLE `training_jobs` ADD `grid_row` varchar(10);