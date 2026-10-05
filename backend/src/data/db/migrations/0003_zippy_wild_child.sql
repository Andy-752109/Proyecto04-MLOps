CREATE TABLE `training_jobs` (
	`id` varchar(64) NOT NULL,
	`status` enum('queued','running','completed','failed','cancelled') NOT NULL DEFAULT 'queued',
	`progress` double NOT NULL DEFAULT 0,
	`config` json NOT NULL,
	`dataset_release` varchar(100) NOT NULL,
	`manifest_id` varchar(100) NOT NULL,
	`mlflow_run_id` varchar(100),
	`error` text,
	`logs` json NOT NULL,
	`heartbeat_at` timestamp,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	`updated_at` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `training_jobs_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE INDEX `training_jobs_status_idx` ON `training_jobs` (`status`);--> statement-breakpoint
CREATE INDEX `training_jobs_heartbeat_at_idx` ON `training_jobs` (`heartbeat_at`);