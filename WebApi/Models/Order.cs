using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace stajprojem.Models
{
    [Table("orders")]
    public class Order
    {
        [Key]
        [Column("id")]
        public int id { get; set; }

        [Column("customer_id")]
        public int? customer_id { get; set; }

        [Column("employee_id")]
        public int? employee_id { get; set; }

        [Column("order_date")]
        public DateTime? order_date { get; set; }

        [Column("total_amount")]
        public decimal? total_amount { get; set; }

        [Column("status")]
        public string? status { get; set; }

        [Column("notes")]
        public string? notes { get; set; }
    }
}
