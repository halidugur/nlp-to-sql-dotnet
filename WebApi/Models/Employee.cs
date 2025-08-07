using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace stajprojem.Models
{
    [Table("employees")]
    public class Employee
    {
        [Key]
        [Column("id")]
        public int id { get; set; }

        [Column("first_name")]
        public string? first_name { get; set; }

        [Column("last_name")]
        public string? last_name { get; set; }

        [Column("department")]
        public string? department { get; set; }

        [Column("position")]
        public string? position { get; set; }

        [Column("salary")]
        public decimal? salary { get; set; }

        [Column("hire_date")]
        public DateTime? hire_date { get; set; }

        [Column("email")]
        public string? email { get; set; }

        [Column("phone")]
        public string? phone { get; set; }
    }
}
