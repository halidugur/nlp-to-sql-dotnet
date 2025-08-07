using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace stajprojem.Models
{
    [Table("customers")] // Tablo ismi küçük harfle
    public class Customer
    {
        [Key]
        [Column("id")]
        public int id { get; set; }

        [Column("company_name")]
        public string? company_name { get; set; }

        [Column("contact_person")]
        public string? contact_person { get; set; }

        [Column("email")]
        public string? email { get; set; }

        [Column("phone")]
        public string? phone { get; set; }

        [Column("address")]
        public string? address { get; set; }

        [Column("city")]
        public string? city { get; set; }

        [Column("created_date")]
        public DateTime? created_date { get; set; }
    }
}
