CREATE TABLE `inference_submissions` (
	`id` bigint unsigned AUTO_INCREMENT NOT NULL,
	`image_id` bigint unsigned NOT NULL,
	`predicted_label` varchar(50) NOT NULL,
	`probabilities` json NOT NULL,
	`model_version` varchar(50) NOT NULL,
	`checkpoint_sha256` varchar(64) NOT NULL,
	`created_at` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `inference_submissions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
ALTER TABLE `inference_submissions` ADD CONSTRAINT `inference_submissions_image_id_images_id_fk` FOREIGN KEY (`image_id`) REFERENCES `images`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX `inference_submissions_image_id_idx` ON `inference_submissions` (`image_id`);