/*
 Navicat Premium Dump SQL

 Source Server         : newera
 Source Server Type    : MySQL
 Source Server Version : 101110 (10.11.10-MariaDB-log)
 Source Host           : 202.155.90.20:3306
 Source Schema         : newera

 Target Server Type    : MySQL
 Target Server Version : 101110 (10.11.10-MariaDB-log)
 File Encoding         : 65001

 Date: 29/11/2025 21:18:00
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for cg_futures_basis_history
-- ----------------------------
DROP TABLE IF EXISTS `cg_futures_basis_history`;
CREATE TABLE `cg_futures_basis_history` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `exchange` varchar(50) NOT NULL,
  `pair` varchar(50) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `open_basis` decimal(18,8) DEFAULT NULL,
  `close_basis` decimal(18,8) DEFAULT NULL,
  `open_change` decimal(18,8) DEFAULT NULL,
  `close_change` decimal(18,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_exchange_pair_interval_time` (`exchange`,`pair`,`interval`,`time`),
  KEY `idx_exchange` (`exchange`),
  KEY `idx_pair` (`pair`),
  KEY `idx_time` (`time`)
) ENGINE=InnoDB AUTO_INCREMENT=319694460 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Table structure for cg_futures_footprint_history
-- ----------------------------
DROP TABLE IF EXISTS `cg_futures_footprint_history`;
CREATE TABLE `cg_futures_footprint_history` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `exchange` varchar(50) NOT NULL,
  `symbol` varchar(50) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `price_start` decimal(20,8) DEFAULT NULL,
  `price_end` decimal(20,8) DEFAULT NULL,
  `taker_buy_volume` decimal(38,8) DEFAULT NULL,
  `taker_sell_volume` decimal(38,8) DEFAULT NULL,
  `taker_buy_volume_usd` decimal(38,8) DEFAULT NULL,
  `taker_sell_volume_usd` decimal(38,8) DEFAULT NULL,
  `taker_buy_trades` int(11) DEFAULT NULL,
  `taker_sell_trades` int(11) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_exchange_symbol_interval_time_price` (`exchange`,`symbol`,`interval`,`time`,`price_start`,`price_end`),
  KEY `idx_exchange` (`exchange`),
  KEY `idx_symbol` (`symbol`),
  KEY `idx_interval` (`interval`),
  KEY `idx_time` (`time`)
) ENGINE=InnoDB AUTO_INCREMENT=834227924 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Table structure for cg_option_exchange_oi_history
-- ----------------------------
DROP TABLE IF EXISTS `cg_option_exchange_oi_history`;
CREATE TABLE `cg_option_exchange_oi_history` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `symbol` varchar(20) NOT NULL,
  `unit` varchar(10) NOT NULL,
  `range` varchar(10) NOT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_symbol_unit_range` (`symbol`,`unit`,`range`),
  KEY `idx_symbol` (`symbol`),
  KEY `idx_unit` (`unit`),
  KEY `idx_range` (`range`)
) ENGINE=InnoDB AUTO_INCREMENT=64005 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Table structure for cg_option_exchange_oi_history_exchange_data
-- ----------------------------
DROP TABLE IF EXISTS `cg_option_exchange_oi_history_exchange_data`;
CREATE TABLE `cg_option_exchange_oi_history_exchange_data` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `option_exchange_oi_history_id` bigint(20) NOT NULL,
  `timestamp_index` int(11) NOT NULL,
  `exchange` varchar(50) NOT NULL,
  `open_interest` decimal(30,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_history_id_timestamp_exchange` (`option_exchange_oi_history_id`,`timestamp_index`,`exchange`),
  KEY `idx_option_exchange_oi_history_id` (`option_exchange_oi_history_id`),
  KEY `idx_exchange` (`exchange`),
  KEY `idx_timestamp_index` (`timestamp_index`),
  CONSTRAINT `cg_option_exchange_oi_history_exchange_data_ibfk_1` FOREIGN KEY (`option_exchange_oi_history_id`) REFERENCES `cg_option_exchange_oi_history` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=20481 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Table structure for cg_option_exchange_oi_history_time_list
-- ----------------------------
DROP TABLE IF EXISTS `cg_option_exchange_oi_history_time_list`;
CREATE TABLE `cg_option_exchange_oi_history_time_list` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `option_exchange_oi_history_id` bigint(20) NOT NULL,
  `timestamp_index` int(11) NOT NULL,
  `timestamp` bigint(20) NOT NULL,
  `price` decimal(20,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_history_id_timestamp_index` (`option_exchange_oi_history_id`,`timestamp_index`),
  KEY `idx_option_exchange_oi_history_id` (`option_exchange_oi_history_id`),
  KEY `idx_timestamp` (`timestamp`),
  CONSTRAINT `cg_option_exchange_oi_history_time_list_ibfk_1` FOREIGN KEY (`option_exchange_oi_history_id`) REFERENCES `cg_option_exchange_oi_history` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=4097 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- ----------------------------
-- Table structure for cg_spot_aggregated_ask_bids_history
-- ----------------------------
DROP TABLE IF EXISTS `cg_spot_aggregated_ask_bids_history`;
CREATE TABLE `cg_spot_aggregated_ask_bids_history` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `exchange_name` varchar(50) NOT NULL,
  `symbol` varchar(20) NOT NULL,
  `base_asset` varchar(20) NOT NULL,
  `interval` varchar(10) NOT NULL,
  `range_percent` varchar(10) NOT NULL,
  `time` bigint(20) NOT NULL,
  `aggregated_bids_usd` decimal(38,8) DEFAULT NULL,
  `aggregated_bids_quantity` decimal(38,8) DEFAULT NULL,
  `aggregated_asks_usd` decimal(38,8) DEFAULT NULL,
  `aggregated_asks_quantity` decimal(38,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_exchange_name_symbol_interval_range_time` (`exchange_name`,`symbol`,`interval`,`range_percent`,`time`),
  KEY `idx_exchange_name` (`exchange_name`),
  KEY `idx_symbol` (`symbol`),
  KEY `idx_base_asset` (`base_asset`),
  KEY `idx_interval` (`interval`),
  KEY `idx_time` (`time`)
) ENGINE=InnoDB AUTO_INCREMENT=489621722 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

SET FOREIGN_KEY_CHECKS = 1;
