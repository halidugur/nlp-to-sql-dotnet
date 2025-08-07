using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace stajprojem.Models
{
    [Table("purchase_orders")]
    public class PurchaseOrder
    {
        [Key]
        [Column("id")]
        public int id { get; set; }

        [Column("supplier_id")]
        public int? supplier_id { get; set; }

        [Column("employee_id")]
        public int? employee_id { get; set; }

        [Column("order_date")]
        public DateTime? order_date { get; set; }

        [Column("total_amount")]
        public decimal? total_amount { get; set; }

        [Column("status")]
        public string? status { get; set; }

        [Column("delivery_date")]
        public DateTime? delivery_date { get; set; }

        [Column("notes")]
        public string? notes { get; set; }
    }
}
